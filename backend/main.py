from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import shutil
import json
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
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

nltk.download('punkt')

app = FastAPI()

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

class SearchRequest(BaseModel):
    query: str
    file_type: Optional[str] = None
    top_k: int = 5

class DirectoryWatchRequest(BaseModel):
    path: str

class TechBrainDemo:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qa_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
        self.documents = {}
        self.embeddings = {}
        self.metadata = {}
        self.watched_directories = set()
        self.observer = watchdog.observers.Observer()
        os.makedirs("uploads", exist_ok=True)

    def extract_text(self, file_path: str, file_type: str) -> str:
        try:
            if file_type == "text":
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            elif file_type == "pdf":
                reader = PdfReader(file_path)
                return " ".join([page.extract_text() for page in reader.pages])
            elif file_type == "document":
                doc = Document(file_path)
                return " ".join([paragraph.text for paragraph in doc.paragraphs])
            elif file_type == "spreadsheet":
                df = pd.read_excel(file_path)
                return df.to_string()
            else:
                return ""
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return ""

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

    def process_file(self, file_path: str, filename: str) -> Dict:
        file_type = self.get_file_type(filename)
        content = self.extract_text(file_path, file_type)
        
        doc_id = str(len(self.documents))
        self.documents[doc_id] = content
        self.embeddings[doc_id] = self.compute_embedding(content)
        
        metadata = {
            "filename": filename,
            "file_type": file_type,
            "upload_time": datetime.now().isoformat(),
            "file_size": os.path.getsize(file_path)
        }
        self.metadata[doc_id] = metadata
        
        return {"doc_id": doc_id, "metadata": metadata}

    def compute_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text)

    def semantic_search(self, query: str, file_type: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        query_embedding = self.compute_embedding(query)
        results = []
        
        for doc_id, doc_embedding in self.embeddings.items():
            if file_type and self.metadata[doc_id]["file_type"] != file_type:
                continue
                
            similarity = np.dot(query_embedding, doc_embedding)
            results.append({
                "doc_id": doc_id,
                "content": self.documents[doc_id][:200] + "...",
                "similarity": float(similarity),
                "metadata": self.metadata[doc_id]
            })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def answer_question(self, question: str, context: str) -> Dict:
        inputs = self.qa_tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.qa_model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)
        
        answer = self.qa_tokenizer.convert_tokens_to_string(
            self.qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end+1])
        )
        
        return {
            "answer": answer,
            "confidence": float(torch.max(outputs.start_logits))
        }

    def watch_directory(self, path: str):
        if path in self.watched_directories:
            return
            
        class FileHandler(watchdog.events.FileSystemEventHandler):
            def __init__(self, tech_brain):
                self.tech_brain = tech_brain
                
            def on_created(self, event):
                if not event.is_directory:
                    self.tech_brain.process_file(event.src_path, os.path.basename(event.src_path))
        
        self.watched_directories.add(path)
        self.observer.schedule(FileHandler(self), path, recursive=False)
        if not self.observer.is_alive():
            self.observer.start()

techbrain = TechBrainDemo()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = techbrain.process_file(file_path, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(request: SearchRequest):
    try:
        results = techbrain.semantic_search(request.query, request.file_type, request.top_k)
        
        # Generate an answer from the most relevant document
        answer = None
        if results:
            answer = techbrain.answer_question(request.query, results[0]["content"])["answer"]
            
        return {
            "answer": answer,
            "relevant_docs": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/watch_directory")
async def watch_directory(request: DirectoryWatchRequest):
    try:
        techbrain.watch_directory(request.path)
        return {"message": f"Now watching directory: {request.path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)