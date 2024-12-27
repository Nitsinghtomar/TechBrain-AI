from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Set
import os
import shutil
import json
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import numpy as np
import uvicorn
from pathlib import Path
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import watchdog.observers
import watchdog.events
import threading
import logging
from collections import defaultdict
import re
import concurrent.futures
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TechBrain AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileMetadata(BaseModel):
    filename: str
    file_type: str
    upload_time: str
    file_size: int
    path: str
    tags: List[str] = []
    summary: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    file_type: Optional[str] = None
    top_k: int = 5
    include_summaries: bool = False
    filter_tags: List[str] = []

class DirectoryWatchRequest(BaseModel):
    path: str
    recursive: bool = True

class TagUpdateRequest(BaseModel):
    doc_id: str
    tags: List[str]

class TechBrainDemo:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qa_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
        self.summarizer = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-cnn')
        self.documents = {}
        self.embeddings = {}
        self.metadata = {}
        self.index = {}  # Inverted index for keyword search
        self.watched_directories = set()
        self.observer = watchdog.observers.Observer()
        self.file_locks = defaultdict(threading.Lock)
        self.upload_folder = Path("uploads")
        self.upload_folder.mkdir(exist_ok=True)
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        
    def preprocess_text(self, text: str) -> str:
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text.lower()

    def extract_text(self, file_path: str, file_type: str) -> str:
        try:
            if file_type == "text":
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            elif file_type == "pdf":
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            elif file_type == "document":
                doc = Document(file_path)
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            elif file_type == "spreadsheet":
                df = pd.read_excel(file_path) if file_path.endswith(('.xlsx', '.xls')) else pd.read_csv(file_path)
                return df.to_string()
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

    def get_file_type(self, filename: str) -> str:
        ext = filename.lower().split('.')[-1]
        type_mapping = {
            'txt': 'text',
            'pdf': 'pdf',
            'doc': 'document',
            'docx': 'document',
            'xls': 'spreadsheet',
            'xlsx': 'spreadsheet',
            'csv': 'spreadsheet',
            'jpg': 'image',
            'png': 'image',
            'dwg': 'cad',
            'dxf': 'cad'
        }
        return type_mapping.get(ext, 'unknown')

    def generate_summary(self, text: str) -> str:
        # Split text into chunks if it's too long
        max_length = 1024
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        summaries = []
        
        for chunk in chunks:
            inputs = self.qa_tokenizer(chunk, return_tensors="pt", max_length=max_length, truncation=True)
            summary_ids = self.summarizer.generate(inputs["input_ids"], max_length=130, min_length=30)
            summary = self.qa_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        
        return " ".join(summaries)

async def process_file(self, background_tasks: BackgroundTasks, file_path: str, filename: str) -> Dict:
    file_type = self.get_file_type(filename)
    
    with self.file_locks[filename]:
        try:
            content = self.extract_text(file_path, file_type)
            processed_content = self.preprocess_text(content)
            
            doc_id = str(len(self.documents))
            self.documents[doc_id] = content
            
            # Compute embeddings in background
            background_tasks.add_task(self._compute_embeddings, doc_id, processed_content)
            
            # Generate summary in background
            background_tasks.add_task(self._generate_summary, doc_id, content)
            
            # Update inverted index
            words = word_tokenize(processed_content)
            for word in words:
                if word not in self.index:
                    self.index[word] = set()
                self.index[word].add(doc_id)
            
            metadata = {
                "filename": filename,
                "file_type": file_type,
                "upload_time": datetime.now().isoformat(),
                "file_size": os.path.getsize(file_path),
                "path": str(file_path),
                "tags": [],
                "summary": None  # Will be updated in background
            }
            self.metadata[doc_id] = metadata
            
            return {"doc_id": doc_id, "metadata": metadata}
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

    async def _compute_embeddings(self, doc_id: str, processed_content: str):
        try:
            self.embeddings[doc_id] = self.embedding_model.encode(processed_content)
        except Exception as e:
            logger.error(f"Error computing embeddings for doc_id {doc_id}: {e}")

    async def _generate_summary(self, doc_id: str, content: str):
        try:
            summary = self.generate_summary(content)
            self.metadata[doc_id]["summary"] = summary
        except Exception as e:
            logger.error(f"Error generating summary for doc_id {doc_id}: {e}")

    def semantic_search(self, query: str, file_type: Optional[str] = None, top_k: int = 5,
                       include_summaries: bool = False, filter_tags: List[str] = []) -> List[Dict]:
        try:
            query_embedding = self.embedding_model.encode(query)
            results = []
            
            for doc_id, doc_embedding in self.embeddings.items():
                if file_type and self.metadata[doc_id]["file_type"] != file_type:
                    continue
                    
                if filter_tags and not set(filter_tags).issubset(set(self.metadata[doc_id]["tags"])):
                    continue
                
                similarity = np.dot(query_embedding, doc_embedding)
                result = {
                    "doc_id": doc_id,
                    "content": self.documents[doc_id][:500] + "...",
                    "similarity": float(similarity),
                    "metadata": self.metadata[doc_id]
                }
                
                if include_summaries:
                    result["summary"] = self.metadata[doc_id].get("summary")
                
                results.append(result)
            
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    def answer_question(self, question: str, context: str) -> Dict:
        try:
            inputs = self.qa_tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.qa_model(**inputs)
            
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits)
            
            answer = self.qa_tokenizer.convert_tokens_to_string(
                self.qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end+1])
            )
            
            confidence = float(torch.max(outputs.start_logits))
            
            # Validate answer
            if not answer or len(answer.split()) < 2:
                return {
                    "answer": "I couldn't find a specific answer to your question in the context.",
                    "confidence": 0.0
                }
            
            return {
                "answer": answer,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error in question answering: {e}")
            raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

    def watch_directory(self, path: str, recursive: bool = True):
        if path in self.watched_directories:
            return
            
        class FileHandler(watchdog.events.FileSystemEventHandler):
            def __init__(self, tech_brain):
                self.tech_brain = tech_brain
                
            def on_created(self, event):
                if not event.is_directory:
                    background_tasks = BackgroundTasks()
                    asyncio.run(self.tech_brain.process_file(
                        event.src_path,
                        os.path.basename(event.src_path),
                        background_tasks
                    ))
        
        self.watched_directories.add(path)
        self.observer.schedule(FileHandler(self), path, recursive=recursive)
        if not self.observer.is_alive():
            self.observer.start()

techbrain = TechBrainDemo()

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        file_path = techbrain.upload_folder / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = await techbrain.process_file(background_tasks, str(file_path), file.filename)
        return result
    except Exception as e:
        logger.error(f"Error in file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(request: SearchRequest):
    try:
        results = techbrain.semantic_search(
            request.query,
            request.file_type,
            request.top_k,
            request.include_summaries,
            request.filter_tags
        )
        
        answer = None
        if results:
            answer = techbrain.answer_question(request.query, results[0]["content"])
            
        return {
            "answer": answer["answer"] if answer else None,
            "confidence": answer["confidence"] if answer else 0.0,
            "relevant_docs": results
        }
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/watch_directory")
async def watch_directory(request: DirectoryWatchRequest):
    try:
        techbrain.watch_directory(request.path, request.recursive)
        return {"message": f"Now watching directory: {request.path}"}
    except Exception as e:
        logger.error(f"Error in directory watch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_tags")
async def update_tags(request: TagUpdateRequest):
    try:
        if request.doc_id not in techbrain.metadata:
            raise HTTPException(status_code=404, detail="Document not found")
            
        techbrain.metadata[request.doc_id]["tags"] = request.tags
        return {"message": "Tags updated successfully"}
    except Exception as e:
        logger.error(f"Error in tag update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)