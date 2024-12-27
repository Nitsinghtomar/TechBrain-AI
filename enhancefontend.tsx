import React, { useState, useEffect, useCallback } from 'react';
import { Upload, Search, File, FolderOpen, AlertCircle, Tag, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';

interface FileInfo {
  id: string;
  name: string;
  type: string;
  uploadTime: string;
  size: number;
  tags: string[];
  summary?: string;
}

interface SearchResult {
  doc_id: string;
  content: string;
  similarity: number;
  summary?: string;
  metadata: {
    filename: string;
    file_type: string;
    upload_time: string;
    file_size: number;
    tags: string[];
    path: string;
  };
}

interface SearchResponse {
  answer: string | null;
  confidence: number;
  relevant_docs: SearchResult[];
}

const API_BASE_URL = 'http://localhost:8000';

const TechBrainDemo: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'search' | 'management'>('search');
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [answer, setAnswer] = useState<{ text: string; confidence: number } | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState<string>('');
  const [watchedDirectories, setWatchedDirectories] = useState<string[]>([]);
  const [fileTypeFilter, setFileTypeFilter] = useState<string>('all');
  const [processingProgress, setProcessingProgress] = useState<number>(0);
  const [uploadSuccess, setUploadSuccess] = useState<boolean>(false);
  const [showAdvancedSearch, setShowAdvancedSearch] = useState<boolean>(false);
  const [includeSummaries, setIncludeSummaries] = useState<boolean>(false);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [expandedResults, setExpandedResults] = useState<Set<string>>(new Set());

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
          size: result.metadata.file_size,
          tags: result.metadata.tags || [],
          summary: result.metadata.summary
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
          include_summaries: includeSummaries,
          filter_tags: selectedTags,
          top_k: 10
        }),
      });

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data: SearchResponse = await response.json();
      setAnswer(data.answer ? { text: data.answer, confidence: data.confidence } : null);
      setSearchResults(data.relevant_docs);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleAddWatchDirectory = async () => {
    const path = window.prompt('Enter directory path to watch:');
    if (!path) return;

    try {
      const response = await fetch(`${API_BASE_URL}/watch_directory`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ path, recursive: true }),
      });

      if (!response.ok) {
        throw new Error('Failed to watch directory');
      }

      setWatchedDirectories(prev => [...prev, path]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to watch directory');
    }
  };

  const handleUpdateTags = async (docId: string, newTags: string[]) => {
    try {
      const response = await fetch(`${API_BASE_URL}/update_tags`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          doc_id: docId,
          tags: newTags
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to update tags');
      }

      setFiles(prev =>
        prev.map(file =>
          file.id === docId ? { ...file, tags: newTags } : file
        )
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update tags');
    }
  };

  const toggleResultExpansion = (docId: string) => {
    setExpandedResults(prev => {
      const newSet = new Set(prev);
      if (newSet.has(docId)) {
        newSet.delete(docId);
      } else {
        newSet.add(docId);
      }
      return newSet;
    });
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
          <div className="space-y-4">
            <div className="flex gap-4">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask any question about your technical data..."
                className="flex-1 p-2 border rounded"
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              />
              <button
                onClick={() => setShowAdvancedSearch(!showAdvancedSearch)}
                className="px-4 py-2 text-gray-600 border rounded hover:bg-gray-50"
              >
                {showAdvancedSearch ? <ChevronUp /> : <ChevronDown />}
              </button>
              <button
                onClick={handleSearch}
                disabled={loading || !query.trim()}
                className="flex items-center gap-2 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
              >
                <Search className="w-5 h-5" />
                Search
              </button>
            </div>

            {showAdvancedSearch && (
              <div className="p-4 bg-gray-50 rounded-lg space-y-4">
                <div className="flex gap-4 items-center">
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
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="includeSummaries"
                      checked={includeSummaries}
                      onChange={(e) => setIncludeSummaries(e.target.checked)}
                      className="rounded"
                    />
                    <label htmlFor="includeSummaries">Include Summaries</label>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {availableTags.map(tag => (
                    <button
                      key={tag}
                      onClick={() => setSelectedTags(prev =>
                        prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
                      )}
                      className={`px-2 py-1 rounded-full text-sm ${
                        selectedTags.includes(tag)
                          ? 'bg-blue-500 text-white'
                          : 'bg-gray-200 text-gray-700'
                      }`}
                    >
                      {tag}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {loading && (
            <div className="w-full bg-gray-200 rounded h-2 my-4">
              <div 
                className="bg-blue-500 h-2 rounded"
                style={{ width: `${processingProgress}%` }}
              />
            </div>
          )}

          {answer && (
            <div className="my-6 bg-blue-50 p-4 rounded-lg">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-bold mb-2">Answer</h3>
                  <p className="text-lg">{answer.text}</p>
                </div>
                <div className="bg-blue-100 px-2 py-1 rounded">
                  {(answer.confidence * 100).toFixed(1)}% confidence
                </div>
              </div>
            </div>
          )}

          <div className="space-y-4 mt-6">
            {searchResults.map((result) => (
              <div key={result.doc_id} className="bg-gray-50 p-4 rounded-lg">
                <div className="flex items-center justify-between cursor-pointer"
                     onClick={() => toggleResultExpansion(result.doc_id)}>
                  <div className="flex items-center gap-2">
                    <File className="w-5 h-5 text-gray-600" />
                    <h4 className="font-medium">{result.metadata.filename}</h4>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-sm bg-blue-100 px-2 py-1 rounded">
                      {(result.similarity * 100).toFixed(1)}% Match
                    </span>
                    {expandedResults.has(result.doc_id) ? <ChevronUp /> : <ChevronDown />}
                  </div>
                </div>

                {expandedResults.has(result.doc_id) && (
                  <div className="mt-4 space-y-4">
                    {result.summary && (
                      <div className="bg-white p-3 rounded">
                        <h5 className="font-medium mb-2">Summary</h5>
                        <p className="text-sm text-gray-600">{result.summary}</p>
                      </div>
                    )}
                    <div className="bg-white p-3 rounded">
                      <h5 className="font-medium mb-2">Content Preview</h5>
                      <p className="text-sm text-gray-600">{result.content}</p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {result.metadata.tags.map(tag => (
                        <span key={tag} className="bg-gray-200 px-2 py-1 rounded-full text-sm">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <div className="text-sm text-gray-500">
                      <p>Path: {result.metadata.path}</p>
                      <p>Uploaded: {formatDate(result.metadata.upload_time)}</p>
                      <p>Size: {formatFileSize(result.metadata.file_size)}</p>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'management' && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex gap-4 mb-6">
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
            <button
              onClick={handleAddWatchDirectory}
              className="flex items-center gap-2 bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
            >
              <FolderOpen className="w-5 h-5" />
              Watch Directory
            </button>
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

          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-3">Watched Directories</h3>
              <div className="space-y-2">
                {watchedDirectories.map((dir, index) => (
                  <div key={index} className="flex items-center gap-2 p-2 bg-gray-50 rounded">
                    <FolderOpen className="w-4 h-4 text-gray-600" />
                    <span className="flex-1">{dir}</span>
                    <RefreshCw className="w-4 h-4 text-gray-400" />
                  </div>
                ))}
                {watchedDirectories.length === 0 && (
                  <p className="text-gray-500 italic">No directories being watched</p>
                )}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Managed Files</h3>
              <div className="space-y-2">
                {files.map((file) => (
                  <div key={file.id} className="p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <File className="w-4 h-4 text-gray-600" />
                      <span className="flex-1 font-medium">{file.name}</span>
                      <span className="text-sm text-gray-500">{file.type}</span>
                      <span className="text-sm text-gray-500">{formatFileSize(file.size)}</span>
                    </div>
                    
                    {file.summary && (
                      <div className="mb-2 text-sm text-gray-600">
                        <p className="font-medium mb-1">Summary:</p>
                        <p>{file.summary}</p>
                      </div>
                    )}
                    
                    <div className="flex flex-wrap items-center gap-2">
                      <Tag className="w-4 h-4 text-gray-500" />
                      {file.tags.map((tag, tagIndex) => (
                        <span
                          key={tagIndex}
                          className="bg-gray-200 px-2 py-1 rounded-full text-sm"
                        >
                          {tag}
                        </span>
                      ))}
                      <button
                        onClick={() => {
                          const newTag = window.prompt('Add new tag:');
                          if (newTag && !file.tags.includes(newTag)) {
                            handleUpdateTags(file.id, [...file.tags, newTag]);
                          }
                        }}
                        className="text-blue-500 text-sm hover:text-blue-600"
                      >
                        + Add Tag
                      </button>
                    </div>
                    
                    <div className="mt-2 text-sm text-gray-500">
                      Uploaded: {formatDate(file.uploadTime)}
                    </div>
                  </div>
                ))}
                {files.length === 0 && (
                  <p className="text-gray-500 italic">No files uploaded yet</p>
                )}
              </div>
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