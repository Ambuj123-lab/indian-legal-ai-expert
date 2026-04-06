import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { FiSend, FiThumbsUp, FiThumbsDown, FiChevronDown, FiChevronUp, FiTrash2, FiCopy, FiMenu } from 'react-icons/fi';
import { useAuth } from '../context/AuthContext';
import { streamChat, submitFeedback, getHistory, clearHistory } from '../api';

export default function Chat({ sidebarOpen, setSidebarOpen, isMobile }) {
    const { user } = useAuth();
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);
    const [imgError, setImgError] = useState(false);
    const [expandedSources, setExpandedSources] = useState({});
    const [toast, setToast] = useState(null);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    useEffect(() => {
        setImgError(false);
    }, [user]);

    useEffect(() => {
        loadHistory();
    }, []);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    async function loadHistory() {
        try {
            const data = await getHistory();
            if (data.history?.length) {
                setMessages(data.history.map((m, i) => ({
                    id: i,
                    role: m.role,
                    content: m.content,
                    sources: m.sources || [],
                    timestamp: m.timestamp,
                })));
            }
        } catch { }
    }

    async function handleSend(e) {
        e?.preventDefault();
        const question = input.trim();
        if (!question || isStreaming) return;

        const userMsg = { id: Date.now(), role: 'user', content: question, timestamp: new Date().toISOString() };
        const botMsg = { id: Date.now() + 1, role: 'assistant', content: '', sources: [], confidence: 0, streaming: true, timestamp: new Date().toISOString() };

        setMessages(prev => [...prev, userMsg, botMsg]);
        setInput('');
        setIsStreaming(true);

        try {
            await streamChat(
                question,
                (token) => {
                    setMessages(prev => {
                        const updated = [...prev];
                        const last = updated[updated.length - 1];
                        last.content += token;
                        return updated;
                    });
                },
                (meta) => {
                    setMessages(prev => {
                        const updated = [...prev];
                        const last = updated[updated.length - 1];
                        last.sources = meta.sources || [];
                        last.confidence = meta.confidence || 0;
                        last.pii_detected = meta.pii_detected || false;
                        last.pii_entities = meta.pii_entities || [];
                        last.showPiiLogs = false;
                        last.streaming = false;
                        return updated;
                    });
                }
            );
        } catch (err) {
            setMessages(prev => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                last.content = 'Sorry, something went wrong. Please try again.';
                last.streaming = false;
                return updated;
            });
        }
        setIsStreaming(false);
        inputRef.current?.focus();
    }

    async function handleFeedback(msg, rating) {
        const userQ = messages.find((m, i) => m.role === 'user' && messages[i + 1]?.id === msg.id);
        if (userQ) {
            try {
                await submitFeedback(userQ.content, msg.content, rating);
                setToast(rating === 'up' ? '👍 Feedback sent! Thank you' : '👎 Feedback noted. We\'ll improve!');
            } catch {
                setToast('⚠ Could not send feedback');
            }
        } else {
            setToast(rating === 'up' ? '👍 Thanks for the feedback!' : '👎 Feedback noted!');
        }
        setMessages(prev => prev.map(m => m.id === msg.id ? { ...m, feedbackGiven: rating } : m));
        // Auto-dismiss toast
        setTimeout(() => setToast(null), 2500);
    }

    function toggleSources(msgId) {
        setExpandedSources(prev => ({ ...prev, [msgId]: !prev[msgId] }));
    }

    async function handleClear() {
        await clearHistory();
        setMessages([]);
    }

    async function handleCopy(msg) {
        const userQ = messages.find((m, i) => m.role === 'user' && messages[i + 1]?.id === msg.id);
        const questionText = userQ ? userQ.content : 'N/A';
        
        const disclaimer = `⚖️ Constitution of India AI Expert | Engineered by Ambuj Kumar Tripathi\n⚠️ Technical Disclaimer: This response was generated dynamically via an Agentic RAG Architecture utilizing Semantic Parent-Child Chunking and Qdrant Vector Retrieval. As with all LLM-driven systems, this output is probabilistic and intended for educational demonstration only. It does NOT constitute professional legal advice. Always consult a qualified advocate for critical legal matters.`;
        
        const textToCopy = `Question: ${questionText}\n\nAnswer:\n${msg.content}\n\n---\n${disclaimer}`;
        
        try {
            await navigator.clipboard.writeText(textToCopy);
            setToast('📋 Copied to clipboard with disclaimer!');
            setTimeout(() => setToast(null), 2500);
        } catch (err) {
            setToast('⚠ Failed to copy');
            setTimeout(() => setToast(null), 2500);
        }
    }

    return (
        <div className="chat-container">
            {/* Header */}
            <div className="chat-header">
                <div className="chat-header-info">
                    <div className="chat-logo">
                        <img 
                            src="/branding/logo.png" 
                            alt="Constitution of India" 
                            className="chat-logo-img"
                            onError={(e) => { e.target.style.display = 'none' }} 
                        />
                        <div className="logo-text">
                            <h3>AI Legal Expert</h3>
                            <p>Constitution • BNS • BNSS • Consumer • IT Act • Motor Vehicles</p>
                            <div className="navbar-credit-text">
                                Built by <span className="credit-author">Ambuj Kumar Tripathi</span> 
                                <span className="credit-sep"> · </span> 
                                <span className="credit-role">RAG Systems Architect</span> 
                                <span className="credit-sep"> · </span> 
                                <a href="https://ambuj-portfolio-v2.netlify.app" target="_blank" rel="noopener noreferrer" className="credit-link">ambuj-portfolio-v2.netlify.app</a>
                            </div>
                        </div>
                    </div>

                    <div className="header-center">
                        <a 
                            href="https://stats.uptimerobot.com/4tYmSQnuBE" 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            className="infra-status-badge"
                            title="Live Infrastructure Status"
                        >
                            <span className="pulse-dot"></span>
                            <span className="status-text">System Live</span>
                            <span className="infra-label">on Production Infra</span>
                        </a>
                    </div>

                    <div className="header-actions">
                        {isMobile && (
                            <button
                                className="hamburger-btn"
                                onClick={() => setSidebarOpen(!sidebarOpen)}
                                title="Menu"
                            >
                                <FiMenu size={22} />
                            </button>
                        )}
                        <div className="user-profile" title={user?.name || 'User'}>
                            {user?.picture && !imgError ? (
                                <img 
                                    src={user.picture} 
                                    alt="Avatar" 
                                    referrerPolicy="no-referrer"
                                    onError={() => setImgError(true)} 
                                />
                            ) : (
                                <div className="user-initial">
                                    {user?.name?.[0]?.toUpperCase() || <FiUser size={16} />}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Messages */}
            <div className="chat-messages">
                {messages.length === 0 && (
                    <div className="chat-welcome">
                        <div className="welcome-icon">⚖️</div>
                        <h3>Welcome to AI Legal Expert</h3>
                        <p>Ask me about Indian laws, your rights, and legal procedures.</p>
                        <div className="suggestion-chips">
                            {[
                                'What are Fundamental Rights under the Constitution?',
                                'Explain Consumer Protection Act 2019',
                                'What is BNS Section 103 (Murder)?',
                                'Cyber crime laws under IT Act 2000'
                            ].map((q, i) => (
                                <button key={i} className="chip" onClick={() => { setInput(q); }}>
                                    {q}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {messages.map((msg) => (
                    <div key={msg.id} className={`message ${msg.role}`}>
                        {msg.role === 'assistant' && (
                            <div className="bot-avatar">
                                <img src="/branding/logo.png" alt="AI" />
                            </div>
                        )}
                        <div className="message-bubble">
                            {msg.role === 'assistant' ? (
                                <>
                                    <div className="message-content markdown-body">
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                            {msg.content || (msg.streaming ? '' : 'No response')}
                                        </ReactMarkdown>
                                        {msg.streaming && <span className="typing-cursor">▊</span>}
                                    </div>

                                    {/* Actions bar */}
                                    {!msg.streaming && msg.content && (
                                        <div className="message-actions">
                                            <div className="action-left">
                                                {msg.confidence > 0 && (
                                                    <span className={`confidence-badge ${msg.confidence >= 70 ? 'high' : msg.confidence >= 40 ? 'mid' : 'low'}`}>
                                                        {Math.round(msg.confidence)}% confidence
                                                    </span>
                                                )}
                                                {msg.pii_detected && (
                                                    <span
                                                        className="pii-badge clickable"
                                                        onClick={() => {
                                                            setMessages(prev => {
                                                                const updated = [...prev];
                                                                const target = updated.find(m => m.id === msg.id);
                                                                if (target) target.showPiiLogs = !target.showPiiLogs;
                                                                return updated;
                                                            });
                                                        }}
                                                        title="Click to view PII detection logs"
                                                    >
                                                        🔒 PII Protected {msg.showPiiLogs ? '▲' : '▼'}
                                                    </span>
                                                )}
                                                {msg.showPiiLogs && msg.pii_entities?.length > 0 && (
                                                    <div className="pii-logs-dropdown">
                                                        <div className="pii-logs-title">🛡️ Microsoft Presidio Detection Logs</div>
                                                        {msg.pii_entities.map((entity, idx) => (
                                                            <div key={idx} className="pii-log-entry">
                                                                <span className="pii-entity-type">{entity.type}</span>
                                                                <span className="pii-entity-score">{Math.round(entity.score * 100)}% confidence</span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>
                                            <div className="action-right">
                                                <button
                                                    onClick={() => handleCopy(msg)}
                                                    title="Copy Answer"
                                                    className="copy-btn"
                                                    style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.8rem' }}
                                                ><FiCopy size={14} /> <span>Copy</span></button>
                                                <button
                                                    onClick={() => handleFeedback(msg, 'up')}
                                                    className={`feedback-btn ${msg.feedbackGiven === 'up' ? 'active-up' : ''}`}
                                                    title="Helpful"
                                                >👍 <span>Helpful</span></button>
                                                <button
                                                    onClick={() => handleFeedback(msg, 'down')}
                                                    className={`feedback-btn ${msg.feedbackGiven === 'down' ? 'active-down' : ''}`}
                                                    title="Not helpful"
                                                >👎 <span>Not helpful</span></button>
                                            </div>
                                        </div>
                                    )}

                                    {/* Sources */}
                                    {!msg.streaming && msg.sources?.length > 0 && (
                                        <div className="message-sources">
                                            <button className="sources-toggle" onClick={() => toggleSources(msg.id)}>
                                                📚 {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''}
                                                {expandedSources[msg.id] ? <FiChevronUp size={14} /> : <FiChevronDown size={14} />}
                                            </button>
                                            {expandedSources[msg.id] && (
                                                <div className="sources-list">
                                                    {msg.sources.map((src, i) => (
                                                        <div key={i} className="source-card">
                                                            <div className="source-header">
                                                                <span className="source-file">{src.file}</span>
                                                                <span className="source-page">Page {src.page}</span>
                                                                <span className="source-score">{(src.score * 100).toFixed(0)}%</span>
                                                            </div>
                                                            <p className="source-preview">{src.preview}</p>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </>
                            ) : (
                                <div className="message-content">
                                    <p>{msg.content}</p>
                                </div>
                            )}
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form className="chat-input-form" onSubmit={handleSend}>
                <div className="input-wrapper">
                    <input
                        ref={inputRef}
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask about Indian laws, rights, procedures..."
                        disabled={isStreaming}
                        autoFocus
                    />
                    <button type="submit" disabled={!input.trim() || isStreaming} className="send-btn">
                        {isStreaming ? <div className="spinner" /> : <FiSend size={18} />}
                    </button>
                </div>
                <p className="input-disclaimer">AI can make mistakes. Verify important legal information with a qualified professional.</p>
            </form>

            {/* Toast notification */}
            {toast && (
                <div style={{
                    position: 'fixed',
                    bottom: '24px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    background: '#1a1a1a',
                    color: '#ededed',
                    padding: '0.65rem 1.4rem',
                    borderRadius: '12px',
                    fontSize: '0.82rem',
                    fontWeight: 500,
                    zIndex: 1000,
                    boxShadow: '0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(16, 185, 129, 0.15)',
                    animation: 'fadeInUp 0.3s ease-out',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    whiteSpace: 'nowrap'
                }}>
                    {toast}
                </div>
            )}
        </div>
    );
}
