"use client";

import { useState, useEffect, useRef } from 'react';
import { 
  UploadCloud, 
  FileText, 
  CheckCircle2, 
  AlertCircle, 
  Loader2, 
  Database, 
  Search, 
  Sparkles, 
  Layers, 
  HardDrive,
  RefreshCw,
  FileCheck
} from 'lucide-react';
import { getApiBaseUrl } from '@/utils/api';

interface IngestedDoc {
  id: string;
  filename: string;
  chunks_count: number;
  characters_count: number;
  size_bytes: number;
  s3_uploaded: boolean;
  s3_key: string | null;
  timestamp: string;
}

interface SearchResult {
  id: string;
  score: number;
  text: string;
  source: string;
  metadata?: any;
}

export default function DocumentIngest({ userName }: { userName: string }) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [lastUploadedDoc, setLastUploadedDoc] = useState<IngestedDoc | null>(null);
  const [documents, setDocuments] = useState<IngestedDoc[]>([]);

  // Search state for test queries
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[] | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch document history
  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${getApiBaseUrl()}/api/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (err) {
      console.error("Failed to load documents:", err);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setStatusMessage(null);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
      setStatusMessage(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setStatusMessage(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${getApiBaseUrl()}/api/upload-document`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok && data.status === 'success') {
        setStatusMessage({
          type: 'success',
          text: data.message || `Successfully ingested ${selectedFile.name}`
        });
        setLastUploadedDoc(data.document);
        setSelectedFile(null);
        if (fileInputRef.current) fileInputRef.current.value = '';
        fetchDocuments();
      } else {
        setStatusMessage({
          type: 'error',
          text: data.detail || 'Upload failed. Please check backend logs.'
        });
      }
    } catch (err: any) {
      console.error("Upload error:", err);
      setStatusMessage({
        type: 'error',
        text: `Network error: ${err.message || 'Cannot reach backend server.'}`
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleTestSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    setSearchResults(null);

    try {
      const res = await fetch(`${getApiBaseUrl()}/api/search?query=${encodeURIComponent(searchQuery)}&top_k=4`);
      if (res.ok) {
        const data = await res.json();
        setSearchResults(data.results || []);
      } else {
        const err = await res.json();
        alert(err.detail || 'Search failed');
      }
    } catch (err: any) {
      console.error("Search error:", err);
      alert("Failed to search vector database");
    } finally {
      setIsSearching(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div style={{ flex: 1, padding: '30px 40px', overflowY: 'auto', maxHeight: '100vh' }}>
      {/* Header */}
      <div style={{ marginBottom: '30px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
          <Database size={28} color="var(--aurora-blue)" />
          <h1 className="text-aurora" style={{ fontSize: '1.8rem', fontWeight: 700 }}>
            Document Ingestion & Knowledge Base
          </h1>
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
          Upload PDFs and documents to automatically extract text, create 1024-dimensional Hugging Face embeddings (<code style={{ color: 'var(--aurora-green)' }}>BAAI/bge-large-en-v1.5</code>), and index into Pinecone with optional AWS S3 backup.
        </p>
      </div>

      {/* Pipeline Status Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '15px', marginBottom: '30px' }}>
        <div className="glass-panel" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ background: 'rgba(96, 239, 255, 0.1)', padding: '10px', borderRadius: '10px' }}>
            <FileText size={22} color="var(--aurora-blue)" />
          </div>
          <div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Format Support</div>
            <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>PDF, TXT, MD, DOC</div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ background: 'rgba(142, 45, 226, 0.15)', padding: '10px', borderRadius: '10px' }}>
            <Sparkles size={22} color="var(--aurora-purple-real)" />
          </div>
          <div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>HF Model</div>
            <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>bge-large-en-v1.5 (1024d)</div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ background: 'rgba(0, 255, 135, 0.1)', padding: '10px', borderRadius: '10px' }}>
            <Layers size={22} color="var(--aurora-green)" />
          </div>
          <div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Vector Database</div>
            <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>Pinecone Serverless</div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ background: 'rgba(255, 255, 255, 0.05)', padding: '10px', borderRadius: '10px' }}>
            <HardDrive size={22} color="#f59e0b" />
          </div>
          <div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Cloud Storage</div>
            <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>AWS S3 Backup</div>
          </div>
        </div>
      </div>

      {/* Main Upload Area */}
      <div className="glass-panel" style={{ padding: '30px', marginBottom: '30px' }}>
        <h2 style={{ fontSize: '1.2rem', marginBottom: '15px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <UploadCloud size={20} color="var(--aurora-blue)" />
          Upload New Document
        </h2>

        {/* Drop Zone */}
        <div 
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          style={{
            border: isDragOver ? '2px dashed var(--aurora-blue)' : '2px dashed var(--glass-border)',
            background: isDragOver ? 'rgba(96, 239, 255, 0.05)' : 'rgba(255, 255, 255, 0.01)',
            borderRadius: '16px',
            padding: '40px 20px',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all 0.25s ease',
            marginBottom: '20px'
          }}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept=".pdf,.txt,.md,.csv,.json,.doc,.docx"
            style={{ display: 'none' }} 
          />

          <UploadCloud 
            size={48} 
            color={isDragOver ? "var(--aurora-blue)" : "var(--text-secondary)"} 
            style={{ margin: '0 auto 15px', display: 'block', transition: 'transform 0.2s ease', transform: isDragOver ? 'scale(1.1)' : 'scale(1)' }} 
          />

          <p style={{ fontSize: '1.05rem', fontWeight: 600, marginBottom: '6px' }}>
            {selectedFile ? selectedFile.name : "Click to select or drag and drop your document here"}
          </p>

          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Supports PDF, TXT, MD documents (Recommended size up to 25 MB)
          </p>
        </div>

        {/* Selected File Details & Upload Action */}
        {selectedFile && (
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between', 
            background: 'rgba(255, 255, 255, 0.03)', 
            padding: '14px 20px', 
            borderRadius: '10px',
            border: '1px solid var(--glass-border)',
            marginBottom: '20px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <FileText size={24} color="var(--aurora-green)" />
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>{selectedFile.name}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Size: {formatFileSize(selectedFile.size)}
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '10px' }}>
              <button 
                onClick={(e) => { e.stopPropagation(); setSelectedFile(null); }}
                className="btn-secondary"
                disabled={isUploading}
                style={{ padding: '8px 14px', fontSize: '0.85rem' }}
              >
                Clear
              </button>
              <button 
                onClick={handleUpload}
                disabled={isUploading}
                className="btn-primary"
                style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 20px' }}
              >
                {isUploading ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Ingesting...
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    Start Ingestion
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Status Message */}
        {statusMessage && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            padding: '14px 18px',
            borderRadius: '10px',
            background: statusMessage.type === 'success' ? 'rgba(0, 255, 135, 0.1)' : 'rgba(239, 68, 68, 0.1)',
            border: `1px solid ${statusMessage.type === 'success' ? 'rgba(0, 255, 135, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
            color: statusMessage.type === 'success' ? 'var(--aurora-green)' : '#f87171',
            fontSize: '0.9rem'
          }}>
            {statusMessage.type === 'success' ? <CheckCircle2 size={20} /> : <AlertCircle size={20} />}
            <span>{statusMessage.text}</span>
          </div>
        )}
      </div>

      {/* Semantic Search / Retrieval Playground */}
      <div className="glass-panel" style={{ padding: '30px', marginBottom: '30px' }}>
        <h2 style={{ fontSize: '1.2rem', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Search size={20} color="var(--aurora-blue)" />
          Test Semantic Search on Vector DB
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '20px' }}>
          Verify your ingested knowledge immediately. The query will be embedded via Hugging Face and matched against Pinecone vectors.
        </p>

        <form onSubmit={handleTestSearch} style={{ display: 'flex', gap: '12px', marginBottom: '20px' }}>
          <input 
            type="text"
            className="input-glass"
            placeholder="Type a test query (e.g. 'refund policy', 'technical architecture', 'pricing plans')..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{ flex: 1 }}
          />
          <button 
            type="submit" 
            className="btn-primary" 
            disabled={isSearching}
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            {isSearching ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
            Search
          </button>
        </form>

        {searchResults && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between' }}>
              <span>Matches Found: {searchResults.length}</span>
              <span>Metric: Cosine Similarity</span>
            </div>

            {searchResults.length === 0 ? (
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontStyle: 'italic', padding: '15px', textAlign: 'center' }}>
                No similar vectors found. Try another query or upload more documents.
              </p>
            ) : (
              searchResults.map((result, idx) => (
                <div 
                  key={result.id || idx}
                  style={{
                    background: 'rgba(255, 255, 255, 0.02)',
                    border: '1px solid var(--glass-border)',
                    borderRadius: '10px',
                    padding: '16px'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span style={{ fontWeight: 600, color: 'var(--aurora-blue)', fontSize: '0.9rem' }}>
                      📄 Source: {result.source || 'Knowledge Base'}
                    </span>
                    <span style={{ 
                      fontSize: '0.75rem', 
                      background: 'rgba(0, 255, 135, 0.15)', 
                      color: 'var(--aurora-green)', 
                      padding: '3px 8px', 
                      borderRadius: '12px',
                      fontWeight: 600
                    }}>
                      Similarity: {(result.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-primary)', whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>
                    {result.text}
                  </p>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Ingested Documents History */}
      <div className="glass-panel" style={{ padding: '30px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h2 style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <FileCheck size={20} color="var(--aurora-green)" />
            Recent Ingested Documents
          </h2>
          <button 
            onClick={fetchDocuments}
            className="btn-secondary"
            style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px', fontSize: '0.8rem' }}
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </div>

        {documents.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontStyle: 'italic', textAlign: 'center', padding: '20px' }}>
            No documents uploaded yet in this session. Upload your first PDF above!
          </p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {documents.map((doc) => (
              <div 
                key={doc.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '12px 18px',
                  background: 'rgba(255, 255, 255, 0.02)',
                  borderRadius: '10px',
                  border: '1px solid var(--glass-border)'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <FileText size={20} color="var(--aurora-blue)" />
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{doc.filename}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      {formatFileSize(doc.size_bytes)} • {new Date(doc.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span style={{ 
                    fontSize: '0.75rem', 
                    background: 'rgba(96, 239, 255, 0.1)', 
                    color: 'var(--aurora-blue)', 
                    padding: '3px 10px', 
                    borderRadius: '12px',
                    fontWeight: 600
                  }}>
                    {doc.chunks_count} Chunks in Pinecone
                  </span>
                  {doc.s3_uploaded && (
                    <span style={{ 
                      fontSize: '0.75rem', 
                      background: 'rgba(245, 158, 11, 0.15)', 
                      color: '#f59e0b', 
                      padding: '3px 10px', 
                      borderRadius: '12px',
                      fontWeight: 600
                    }}>
                      S3 Backed
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
