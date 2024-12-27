import React, { useState, useEffect } from 'react';
import { Upload, Search, File, FolderOpen, AlertCircle } from 'lucide-react';

interface FileInfo {
  id: string;
  name: string;
  type: string;
  uploadTime: string;
  size: number;
}

interface SearchResult {
  doc_id: string;
  content: string;
  similarity: number;
  metadata: {
    filename: string;
    file_type: string;
    upload_time: string;
    file_size: number;
  };
}

const API_BASE_URL = 'http://localhost:8000';

const TechBrainDemo: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'search' | 'management'>('search');
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [answer, setAnswer] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState<string>('');
  const [watchedDirectories, setWatchedDirectories] = useState<string[]>([]);
  const [fileTypeFilter, setFileTypeFilter] = useState<string>('all');
  const [processingProgress, setProcessingProgress] = useState<number>(0);
  const [uploadSuccess, setUploadSuccess] = useState<boolean>(false);

  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  useEffect(() => {
    if (uploadSuccess) {
      const timer = setTimeout(() => setUploadSuccess(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [uploadSuccess]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFiles = event.target.files;
    if (!uploadedFiles?.length) return;

    setLoading(true);
    setError(null);
    setProcessingProgress(0);

    try {
      for (let i = 0; i < uploadedFiles.length; i++) {
        const file = uploadedFiles[i];
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/upload`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Failed to upload ${file.name}`);
        }

        const result = await response.json();
        
        setFiles(prev => [...prev, {
          id: result.doc_id,
          name: result.metadata.filename,
          type: result.metadata.file_type,
          uploadTime: result.metadata.upload_time,
          size: result.metadata.file_size
        }]);

        setProcessingProgress(((i + 1) / uploadedFiles.length) * 100);
      }

      setUploadSuccess(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          file_type: fileTypeFilter === 'all' ? null : fileTypeFilter,
        }),
      });

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const { answer, relevant_docs } = await response.json();
      setAnswer(answer);
      setSearchResults(relevant_docs);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="container mx-auto p-4 max-w-6xl">
      <h1 className="text-3xl font-bold mb-8">TechBrain AI - Intelligent Data Management</h1>

      <div className="mb-6">
        <div className="border-b border-gray-200">
          <div className="flex gap-4">
            <button
              onClick={() => setActiveTab('search')}
              className={`px-4 py-2 ${activeTab === 'search' ? 'border-b-2 border-blue-500' : ''}`}
            >
              Search & Analysis
            </button>
            <button
              onClick={() => setActiveTab('management')}
              className={`px-4 py-2 ${activeTab === 'management' ? 'border-b-2 border-blue-500' : ''}`}
            >
              File Management
            </button>
          </div>
        </div>
      </div>

      {activeTab === 'search' && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex gap-4 mb-4">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask any question about your technical data..."
              className="flex-1 p-2 border rounded"
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            />
            <select
              value={fileTypeFilter}
              onChange={(e) => setFileTypeFilter(e.target.value)}
              className="p-2 border rounded"
            >
              <option value="all">All Files</option>
              <option value="text">Text Files</option>
              <option value="pdf">PDFs</option>
              <option value="document">Documents</option>
              <option value="spreadsheet">Spreadsheets</option>
            </select>
            <button
              onClick={handleSearch}
              disabled={loading || !query.trim()}
              className="flex items-center gap-2 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
            >
              <Search className="w-5 h-5" />
              Search
            </button>
          </div>

          {loading && (
            <div className="w-full bg-gray-200 rounded h-2 mb-4">
              <div 
                className="bg-blue-500 h-2 rounded"
                style={{ width: `${processingProgress}%` }}
              />
            </div>
          )}

          {answer && (
            <div className="mb-4 bg-blue-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Answer</h3>
              <p className="text-lg">{answer}</p>
            </div>
          )}

          {searchResults.map((result) => (
            <div key={result.doc_id} className="bg-gray-50 p-4 rounded-lg mb-4">
              <div className="flex items-center justify-between">
                <h4 className="font-medium">{result.metadata.filename}</h4>
                <span className="text-sm bg-blue-100 px-2 py-1 rounded">
                  {(result.similarity * 100).toFixed(1)}% Match
                </span>
              </div>
              <p className="text-sm text-gray-600 mt-1">{result.content}</p>
              <p className="text-sm text-gray-500 mt-2">
                Uploaded: {formatDate(result.metadata.upload_time)}
              </p>
            </div>
          ))}
        </div>
      )}

      {activeTab === 'management' && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex gap-4 mb-4">
            <label className="flex items-center gap-2 bg-blue-500 text-white px-4 py-2 rounded cursor-pointer hover:bg-blue-600">
              <Upload className="w-5 h-5" />
              Upload Files
              <input
                type="file"
                className="hidden"
                onChange={handleFileUpload}
                multiple
                accept=".txt,.pdf,.doc,.docx,.xls,.xlsx,.csv"
              />
            </label>
          </div>

          {loading && (
            <div className="mb-4">
              <div className="w-full bg-gray-200 rounded h-2 mb-2">
                <div 
                  className="bg-blue-500 h-2 rounded"
                  style={{ width: `${processingProgress}%` }}
                />
              </div>
              <p className="text-sm text-gray-600">Processing files...</p>
            </div>
          )}

          {uploadSuccess && (
            <div className="mb-4 bg-green-50 border border-green-200 p-4 rounded">
              <p className="text-green-700">Files uploaded successfully!</p>
            </div>
          )}

          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-3">Managed Files</h3>
            <div className="space-y-2">
              {files.map((file) => (
                <div key={file.id} className="flex items-center gap-2 p-2 bg-gray-50 rounded">
                  <File className="w-4 h-4 text-gray-600" />
                  <span className="flex-1">{file.name}</span>
                  <span className="text-sm text-gray-500">{file.type}</span>
                  <span className="text-sm text-gray-500">{formatFileSize(file.size)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-4 bg-red-50 border border-red-200 p-4 rounded flex items-center gap-2 text-red-700">
          <AlertCircle className="w-5 h-5" />
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

export default TechBrainDemo;