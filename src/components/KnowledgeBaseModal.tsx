import React, { useState } from 'react';
import { Database, Plus, Search, RefreshCw, FileText, CheckCircle2, AlertCircle, Layers, Sparkles, X, Upload } from 'lucide-react';
import { DocumentItem } from '../types';

interface KnowledgeBaseModalProps {
  isOpen: boolean;
  onClose: () => void;
  documents: DocumentItem[];
  totalChunks: number;
  onIngestDocument: (doc: { title: string; category: string; source: string; content: string }) => Promise<boolean>;
  onResetCorpus: () => Promise<void>;
  isLoading: boolean;
}

export const KnowledgeBaseModal: React.FC<KnowledgeBaseModalProps> = ({
  isOpen,
  onClose,
  documents,
  totalChunks,
  onIngestDocument,
  onResetCorpus,
  isLoading
}) => {
  const [searchFilter, setSearchFilter] = useState('');
  const [selectedDoc, setSelectedDoc] = useState<DocumentItem | null>(null);
  const [isAddingNew, setIsAddingNew] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newCategory, setNewCategory] = useState('custom_knowledge');
  const [newSource, setNewSource] = useState('User Upload');
  const [newContent, setNewContent] = useState('');
  const [ingestStatus, setIngestStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState('');

  if (!isOpen) return null;

  const filteredDocs = documents.filter(doc => 
    doc.title.toLowerCase().includes(searchFilter.toLowerCase()) ||
    doc.category.toLowerCase().includes(searchFilter.toLowerCase()) ||
    doc.content.toLowerCase().includes(searchFilter.toLowerCase())
  );

  const handleSubmitNew = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim() || !newContent.trim()) return;

    setIngestStatus('submitting');
    try {
      const success = await onIngestDocument({
        title: newTitle.trim(),
        category: newCategory.trim(),
        source: newSource.trim(),
        content: newContent.trim()
      });

      if (success) {
        setIngestStatus('success');
        setStatusMessage(`Successfully indexed and chunked "${newTitle}"!`);
        setNewTitle('');
        setNewContent('');
        setTimeout(() => {
          setIsAddingNew(false);
          setIngestStatus('idle');
        }, 1200);
      } else {
        setIngestStatus('error');
        setStatusMessage('Failed to ingest document. Please verify inputs.');
      }
    } catch (err: any) {
      setIngestStatus('error');
      setStatusMessage(err?.message || 'Error ingesting document');
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      if (text) {
        setNewContent(text);
        if (!newTitle) {
          setNewTitle(file.name.replace(/\.[^/.]+$/, ""));
        }
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fadeIn">
      <div 
        id="knowledge-base-modal" 
        className="bg-neutral-900 border border-neutral-700/80 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden text-neutral-100"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-800 bg-neutral-950/60">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-teal-500/10 border border-teal-500/30 text-teal-400">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-white tracking-tight">Active Knowledge Base & Vector Store</h2>
                <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/40">
                  {documents.length} Docs / {totalChunks} Chunks
                </span>
              </div>
              <p className="text-xs text-neutral-400">
                FAISS Dense Embeddings & BM25 Inverted Index live memory store
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              id="btn-toggle-add-doc"
              onClick={() => {
                setIsAddingNew(!isAddingNew);
                setSelectedDoc(null);
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                isAddingNew
                  ? 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
                  : 'bg-teal-500 text-neutral-950 font-semibold hover:bg-teal-400 shadow-sm'
              }`}
            >
              <Plus className="w-4 h-4" />
              {isAddingNew ? 'View Documents' : 'Ingest New Document'}
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {isAddingNew ? (
            /* Ingest New Document Form */
            <form onSubmit={handleSubmitNew} className="space-y-4 bg-neutral-950/70 p-5 rounded-xl border border-neutral-800">
              <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
                <div className="flex items-center gap-2 text-teal-400 font-semibold text-sm">
                  <Sparkles className="w-4 h-4" />
                  Ingest Custom Knowledge Into Active RAG Corpus
                </div>
                <label className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-md bg-neutral-800 border border-neutral-700 text-neutral-300 hover:text-white cursor-pointer hover:bg-neutral-700 transition-colors">
                  <Upload className="w-3.5 h-3.5 text-teal-400" />
                  Upload Text File
                  <input type="file" accept=".txt,.md,.json,.csv" onChange={handleFileUpload} className="hidden" />
                </label>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-neutral-300 mb-1">Document Title</label>
                  <input
                    id="input-doc-title"
                    type="text"
                    required
                    placeholder="e.g. Modern Neural Attention & Transformers"
                    value={newTitle}
                    onChange={(e) => setNewTitle(e.target.value)}
                    className="w-full px-3 py-2 text-sm bg-neutral-900 border border-neutral-700 rounded-lg text-white placeholder-neutral-500 focus:outline-none focus:border-teal-500"
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs font-medium text-neutral-300 mb-1">Category</label>
                    <input
                      type="text"
                      placeholder="e.g. artificial_intelligence"
                      value={newCategory}
                      onChange={(e) => setNewCategory(e.target.value)}
                      className="w-full px-3 py-2 text-sm bg-neutral-900 border border-neutral-700 rounded-lg text-white placeholder-neutral-500 focus:outline-none focus:border-teal-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-neutral-300 mb-1">Source / Citation</label>
                    <input
                      type="text"
                      placeholder="e.g. Research Paper / Notes"
                      value={newSource}
                      onChange={(e) => setNewSource(e.target.value)}
                      className="w-full px-3 py-2 text-sm bg-neutral-900 border border-neutral-700 rounded-lg text-white placeholder-neutral-500 focus:outline-none focus:border-teal-500"
                    />
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-neutral-300 mb-1">
                  Document Content (Will be automatically split into overlapping vector chunks)
                </label>
                <textarea
                  id="textarea-doc-content"
                  required
                  rows={6}
                  placeholder="Paste multi-paragraph text or documentation here. The RAG pipeline will calculate 128-dimensional dense vector embeddings and BM25 token frequencies..."
                  value={newContent}
                  onChange={(e) => setNewContent(e.target.value)}
                  className="w-full px-3 py-2 text-sm bg-neutral-900 border border-neutral-700 rounded-lg text-white placeholder-neutral-500 focus:outline-none focus:border-teal-500 font-mono text-xs"
                />
              </div>

              {ingestStatus === 'success' && (
                <div className="flex items-center gap-2 p-3 rounded-lg bg-emerald-950/60 border border-emerald-800/80 text-emerald-300 text-xs">
                  <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                  <span>{statusMessage}</span>
                </div>
              )}

              {ingestStatus === 'error' && (
                <div className="flex items-center gap-2 p-3 rounded-lg bg-rose-950/60 border border-rose-800/80 text-rose-300 text-xs">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <span>{statusMessage}</span>
                </div>
              )}

              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setIsAddingNew(false)}
                  className="px-4 py-2 text-xs font-medium text-neutral-400 hover:text-white transition-colors"
                >
                  Cancel
                </button>
                <button
                  id="btn-submit-ingest"
                  type="submit"
                  disabled={ingestStatus === 'submitting' || !newTitle || !newContent}
                  className="flex items-center gap-2 px-5 py-2 rounded-lg text-xs font-semibold bg-teal-500 hover:bg-teal-400 text-neutral-950 disabled:opacity-50 transition-all shadow-md"
                >
                  {ingestStatus === 'submitting' ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Chunking & Vectorizing...
                    </>
                  ) : (
                    <>
                      <Layers className="w-3.5 h-3.5" />
                      Index & Vectorize Document
                    </>
                  )}
                </button>
              </div>
            </form>
          ) : (
            /* Document List & Inspector */
            <div className="space-y-4">
              <div className="flex flex-col sm:flex-row gap-3 items-center justify-between">
                <div className="relative w-full sm:w-80">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-neutral-400" />
                  <input
                    type="text"
                    placeholder="Search documents and topics..."
                    value={searchFilter}
                    onChange={(e) => setSearchFilter(e.target.value)}
                    className="w-full pl-9 pr-3 py-1.5 text-xs bg-neutral-950 border border-neutral-800 rounded-lg text-white placeholder-neutral-500 focus:outline-none focus:border-teal-500"
                  />
                </div>

                <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
                  <button
                    onClick={onResetCorpus}
                    disabled={isLoading}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-neutral-400 hover:text-neutral-200 bg-neutral-950 border border-neutral-800 rounded-lg hover:border-neutral-700 transition-colors"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
                    Reset to Default Corpus
                  </button>
                </div>
              </div>

              {/* Grid of Documents */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-[55vh] overflow-y-auto pr-1">
                {filteredDocs.map((doc) => (
                  <div
                    key={doc.id}
                    onClick={() => setSelectedDoc(selectedDoc?.id === doc.id ? null : doc)}
                    className={`p-4 rounded-xl border transition-all cursor-pointer text-left ${
                      selectedDoc?.id === doc.id
                        ? 'bg-teal-950/30 border-teal-500/60 ring-1 ring-teal-500/40'
                        : 'bg-neutral-950/60 border-neutral-800/80 hover:border-neutral-700 hover:bg-neutral-950'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2 mb-1.5">
                      <h3 className="text-sm font-semibold text-white line-clamp-1">{doc.title}</h3>
                      {doc.isCustom && (
                        <span className="px-1.5 py-0.5 text-[10px] font-bold rounded bg-teal-500/20 text-teal-300 border border-teal-500/40 uppercase tracking-wider flex-shrink-0">
                          Custom
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-2 text-[11px] text-neutral-400 mb-2">
                      <span className="px-1.5 py-0.5 rounded bg-neutral-800/80 font-mono text-[10px] text-neutral-300">
                        {doc.category}
                      </span>
                      <span>•</span>
                      <span className="flex items-center gap-1">
                        <Layers className="w-3 h-3 text-teal-400" />
                        {doc.chunkCount} Chunks
                      </span>
                    </div>

                    <p className="text-xs text-neutral-300/90 line-clamp-2 leading-relaxed">
                      {doc.content}
                    </p>

                    {selectedDoc?.id === doc.id && (
                      <div className="mt-3 pt-3 border-t border-neutral-800/80 text-[11px] text-neutral-400 space-y-1.5">
                        <div className="text-neutral-300 font-mono text-[10px]">Source: {doc.source}</div>
                        <div className="bg-neutral-900 p-2.5 rounded-lg border border-neutral-800 font-mono text-[11px] text-neutral-200 max-h-36 overflow-y-auto whitespace-pre-wrap leading-normal">
                          {doc.content}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-neutral-800 bg-neutral-950/80 flex items-center justify-between text-xs text-neutral-400">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span>Hybrid Inverted Index & FAISS Embeddings active</span>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-neutral-800 hover:bg-neutral-700 text-white rounded-lg text-xs font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
