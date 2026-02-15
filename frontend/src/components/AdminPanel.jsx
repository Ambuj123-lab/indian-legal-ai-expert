import { useState, useEffect, useRef } from 'react';
import { FiRefreshCw, FiTrash2, FiFile, FiCheck, FiAlertCircle, FiUploadCloud, FiLoader, FiCheckCircle, FiX } from 'react-icons/fi';
import { syncDocuments, listDocuments, deleteDocument, uploadTempFile, listTempUploads, deleteTempUpload } from '../api';

export default function AdminPanel({ isAdmin = false }) {
    const [documents, setDocuments] = useState([]);
    const [stats, setStats] = useState({});
    const [syncing, setSyncing] = useState(false);
    const [syncResult, setSyncResult] = useState(null);
    const [showPanel, setShowPanel] = useState(false);

    // Temp upload state
    const [tempFiles, setTempFiles] = useState([]);
    const [uploadState, setUploadState] = useState(null); // null | 'uploading' | 'indexing' | 'done' | 'error'
    const [uploadInfo, setUploadInfo] = useState({});
    const fileInputRef = useRef(null);

    useEffect(() => {
        loadTempFiles();
    }, []);

    useEffect(() => {
        if (showPanel) loadDocuments();
    }, [showPanel]);

    async function loadDocuments() {
        try {
            const data = await listDocuments();
            setDocuments(data.documents || []);
            setStats({
                total_documents: data.total_documents,
                total_chunks: data.total_chunks,
                total_parent_chunks: data.total_parent_chunks,
                total_child_chunks: data.total_child_chunks,
            });
        } catch { }
    }

    async function loadTempFiles() {
        try {
            const data = await listTempUploads();
            setTempFiles(data.files || []);
        } catch { }
    }

    async function handleSync() {
        setSyncing(true);
        setSyncResult(null);
        try {
            const data = await syncDocuments();
            setSyncResult(data.results?.summary || {});
            await loadDocuments();
        } catch (err) {
            setSyncResult({ error: 'Sync failed' });
        }
        setSyncing(false);
    }

    async function handleDelete(fileName) {
        if (!confirm(`Delete "${fileName}" and all its vectors?`)) return;
        try {
            await deleteDocument(fileName);
            await loadDocuments();
        } catch { }
    }

    async function handleFileUpload(e) {
        const file = e.target.files?.[0];
        if (!file) return;

        if (!file.name.endsWith('.pdf')) {
            setUploadState('error');
            setUploadInfo({ name: file.name, error: 'Only PDF files are supported' });
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            setUploadState('error');
            setUploadInfo({ name: file.name, error: 'File too large (max 10MB)' });
            return;
        }

        // Stage 1: Uploading
        setUploadState('uploading');
        setUploadInfo({ name: file.name, size: formatBytes(file.size) });

        try {
            // Stage 2: Indexing (backend processes)
            setUploadState('indexing');
            const result = await uploadTempFile(file);

            // Stage 3: Done
            setUploadState('done');
            setUploadInfo({
                name: file.name,
                size: formatBytes(file.size),
                parentChunks: result.parent_chunks,
                childChunks: result.child_chunks,
            });

            await loadTempFiles();

            // Clear done state after 4 seconds
            setTimeout(() => setUploadState(null), 4000);
        } catch (err) {
            setUploadState('error');
            setUploadInfo({ name: file.name, error: err.message || 'Upload failed' });
        }

        fileInputRef.current.value = '';
    }

    async function handleDeleteTemp(fileName) {
        try {
            await deleteTempUpload(fileName);
            await loadTempFiles();
        } catch { }
    }

    function formatBytes(bytes) {
        if (!bytes) return '—';
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }

    return (
        <div className="admin-panel-wrapper">
            {/* ===== YOUR UPLOADS SECTION (Always Visible) ===== */}
            <div className="upload-section">
                <h4 className="section-title">📎 Your Uploads</h4>
                <p className="section-desc">Temporary — auto-removed on logout</p>

                {/* Upload Area */}
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                />
                <button
                    className="upload-area"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploadState === 'uploading' || uploadState === 'indexing'}
                >
                    <FiUploadCloud size={20} />
                    <span>Upload PDF</span>
                    <span className="upload-hint">Max 10MB • PDF only</span>
                </button>

                {/* Upload Progress */}
                {uploadState && (
                    <div className={`upload-progress ${uploadState}`}>
                        {uploadState === 'uploading' && (
                            <>
                                <FiLoader size={14} className="spinning" />
                                <div className="progress-text">
                                    <strong>Uploading {uploadInfo.name}</strong>
                                    <span>{uploadInfo.size}</span>
                                </div>
                            </>
                        )}
                        {uploadState === 'indexing' && (
                            <>
                                <FiLoader size={14} className="spinning" />
                                <div className="progress-text">
                                    <strong>Indexing {uploadInfo.name}...</strong>
                                    <span>Creating parent-child chunks & embedding</span>
                                </div>
                            </>
                        )}
                        {uploadState === 'done' && (
                            <>
                                <FiCheckCircle size={14} />
                                <div className="progress-text">
                                    <strong>✅ Indexed: {uploadInfo.name}</strong>
                                    <span>{uploadInfo.parentChunks} parent • {uploadInfo.childChunks} child chunks</span>
                                </div>
                            </>
                        )}
                        {uploadState === 'error' && (
                            <>
                                <FiAlertCircle size={14} />
                                <div className="progress-text">
                                    <strong>❌ {uploadInfo.name}</strong>
                                    <span>{uploadInfo.error}</span>
                                </div>
                                <button className="dismiss-btn" onClick={() => setUploadState(null)}><FiX size={12} /></button>
                            </>
                        )}
                    </div>
                )}

                {/* Temp File List */}
                {tempFiles.length > 0 && (
                    <div className="temp-file-list">
                        {tempFiles.map((file, i) => (
                            <div key={i} className="temp-file-item">
                                <div className="temp-file-info">
                                    <FiFile size={14} />
                                    <span className="temp-file-name">{file.file_name}</span>
                                    <span className="temp-file-chunks">{file.chunk_count} chunks</span>
                                </div>
                                <button className="temp-file-delete" onClick={() => handleDeleteTemp(file.file_name)} title="Remove">
                                    <FiTrash2 size={12} />
                                </button>
                            </div>
                        ))}
                    </div>
                )}

                {tempFiles.length === 0 && !uploadState && (
                    <p className="empty-state-small">No files uploaded yet</p>
                )}
            </div>

            {/* ===== ADMIN PANEL (Collapsible) ===== */}
            <div className="admin-panel">
                <button className="admin-toggle" onClick={() => setShowPanel(!showPanel)}>
                    {showPanel ? '▼' : '▶'} Core Knowledge Base
                </button>

                {showPanel && (
                    <div className="admin-content">
                        {/* Stats */}
                        <div className="admin-stats">
                            <div className="stat-card">
                                <span className="stat-value">{stats.total_documents || 0}</span>
                                <span className="stat-label">Documents</span>
                            </div>
                            <div className="stat-card">
                                <span className="stat-value">{stats.total_parent_chunks || 0}</span>
                                <span className="stat-label">Parent Chunks</span>
                            </div>
                            <div className="stat-card">
                                <span className="stat-value">{stats.total_child_chunks || 0}</span>
                                <span className="stat-label">Child Chunks</span>
                            </div>
                        </div>

                        {/* Sync Button — Admin Only */}
                        {isAdmin && (
                            <button className="sync-btn" onClick={handleSync} disabled={syncing}>
                                <FiRefreshCw size={16} className={syncing ? 'spinning' : ''} />
                                {syncing ? 'Syncing...' : 'Sync Knowledge Base'}
                            </button>
                        )}

                        {/* Sync Results */}
                        {syncResult && (
                            <div className="sync-results">
                                {syncResult.error ? (
                                    <span className="sync-error"><FiAlertCircle /> {syncResult.error}</span>
                                ) : (
                                    <>
                                        {syncResult.added_count > 0 && <span className="sync-added">+{syncResult.added_count} added</span>}
                                        {syncResult.updated_count > 0 && <span className="sync-updated">↻{syncResult.updated_count} updated</span>}
                                        {syncResult.deleted_count > 0 && <span className="sync-deleted">-{syncResult.deleted_count} deleted</span>}
                                        {syncResult.unchanged_count > 0 && <span className="sync-unchanged">={syncResult.unchanged_count} unchanged</span>}
                                        {syncResult.added_count === 0 && syncResult.updated_count === 0 && syncResult.deleted_count === 0 && (
                                            <span className="sync-unchanged"><FiCheck /> Everything up to date</span>
                                        )}
                                    </>
                                )}
                            </div>
                        )}

                        {/* Document List */}
                        <div className="doc-list">
                            {documents.map((doc, i) => (
                                <div key={i} className={`doc-item ${doc.status}`}>
                                    <div className="doc-info">
                                        <FiFile size={16} />
                                        <span className="doc-name">{doc.file_name}</span>
                                        <span className="doc-meta">{doc.chunk_count} chunks • {formatBytes(doc.file_size)}</span>
                                    </div>
                                    <div className="doc-actions">
                                        <span className={`status-badge ${doc.status}`}>{doc.status}</span>
                                        {isAdmin && doc.status === 'active' && (
                                            <button className="doc-delete" onClick={() => handleDelete(doc.file_name)} title="Delete">
                                                <FiTrash2 size={14} />
                                            </button>
                                        )}
                                    </div>
                                </div>
                            ))}
                            {documents.length === 0 && (
                                <p className="empty-state">No documents indexed. Click "Sync Knowledge Base" to start.</p>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
