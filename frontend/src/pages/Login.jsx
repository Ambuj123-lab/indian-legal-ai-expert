import { useState, useEffect } from 'react';
import { FaReact, FaPython, FaDatabase, FaShieldAlt } from 'react-icons/fa';
import { SiFastapi, SiMongodb, SiSupabase, SiVite } from 'react-icons/si';
import { FiActivity, FiCpu } from 'react-icons/fi';
import { getLoginUrl } from '../api';

const techStack = [
    { icon: <FaReact />, name: "React", color: "#61DAFB" },
    { icon: <SiVite />, name: "Vite", color: "#646CFF" },
    { icon: <FaPython />, name: "Python", color: "#3776AB" },
    { icon: <SiFastapi />, name: "FastAPI", color: "#009688" },
    { icon: <FiCpu />, name: "LangGraph", color: "#A29BFE" },
    { icon: <FaDatabase />, name: "Qdrant", color: "#DC382D" },
    { icon: <SiSupabase />, name: "Supabase", color: "#3ECF8E" },
    { icon: <SiMongodb />, name: "MongoDB", color: "#47A248" },
    { icon: <FaShieldAlt />, name: "Presidio", color: "#0078D4" },
    { icon: <FiActivity />, name: "Langfuse", color: "#F59E0B" },
];

export default function Login() {
    const [width, setWidth] = useState(window.innerWidth);

    useEffect(() => {
        const handleResize = () => setWidth(window.innerWidth);
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const isMobile = width <= 768;
    const isSmallMobile = width <= 480;

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            width: '100%',
            height: '100vh',
            overflowY: isMobile ? 'auto' : 'hidden',
            overflowX: 'hidden',
            background: '#000000'
        }}>
            <style>{`
                @keyframes red-heartbeat-glow {
                    0%   { box-shadow: 0 0 0px rgba(185, 28, 28, 0); border-color: rgba(255, 255, 255, 0.1); }
                    30%  { box-shadow: 0 0 0px rgba(185, 28, 28, 0); border-color: rgba(255, 255, 255, 0.1); }
                    40%  { box-shadow: 0 0 25px rgba(185, 28, 28, 0.8), inset 0 0 8px rgba(153, 27, 27, 0.4); border-color: rgba(185, 28, 28, 0.9); }
                    45%  { box-shadow: 0 0 8px rgba(185, 28, 28, 0.3); border-color: rgba(185, 28, 28, 0.4); }
                    55%  { box-shadow: 0 0 40px rgba(153, 27, 27, 1), inset 0 0 15px rgba(153, 27, 27, 0.8); border-color: #dc2626; }
                    70%  { box-shadow: 0 0 0px rgba(185, 28, 28, 0); border-color: rgba(255, 255, 255, 0.1); }
                    100% { box-shadow: 0 0 0px rgba(185, 28, 28, 0); border-color: rgba(255, 255, 255, 0.1); }
                }
                .status-badge {
                    display: inline-flex; align-items: center; gap: 6px;
                    padding: 4px 12px;
                    background: #000000;
                    animation: red-heartbeat-glow 4s ease-in-out infinite;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 6px; text-decoration: none; color: #ffffff;
                    font-size: 10px; font-weight: 600; letter-spacing: 0.04em;
                    cursor: pointer; white-space: nowrap;
                    transition: border-color 0.3s;
                }
            `}</style>
            {/* ============ TOP AREA: LEFT + RIGHT ============ */}
            <div style={{
                display: 'flex',
                flexDirection: isMobile ? 'column' : 'row',
                flex: 1,
                overflow: isMobile ? 'visible' : 'hidden',
                minHeight: isMobile ? 'auto' : 0
            }}>
                {/* ============ LEFT SIDE ============ */}
                <div style={{
                    width: isMobile ? '100%' : '420px',
                    minWidth: isMobile ? 'auto' : '420px',
                    background: '#0a0a0a',
                    borderRight: isMobile ? 'none' : '1px solid rgba(255,255,255,0.06)',
                    borderBottom: isMobile ? '1px solid rgba(255,255,255,0.06)' : 'none',
                    display: 'flex',
                    flexDirection: 'column',
                    padding: isSmallMobile ? '1.5rem' : isMobile ? '2rem' : '2.5rem',
                    position: 'relative',
                    zIndex: 2,
                    overflowY: 'auto'
                }}>
                    {/* Logo & Status Badge */}
                    <div style={{ marginBottom: isMobile ? '1rem' : '2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                            <img
                                src="/branding/logo.png"
                                alt="Logo"
                                style={{
                                    width: isSmallMobile ? '44px' : '60px',
                                    height: isSmallMobile ? '44px' : '60px',
                                    objectFit: 'contain',
                                    borderRadius: '8px'
                                }}
                                onError={(e) => { e.target.style.display = 'none' }}
                            />
                        </div>
                        <a href="https://stats.uptimerobot.com/4tYmSQnuBE" target="_blank" rel="noreferrer" className="status-badge">
                            <span style={{ position: 'relative', width: '8px', height: '8px', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>
                                <span style={{ position: 'absolute', width: '8px', height: '8px', borderRadius: '50%', background: 'rgba(185, 28, 28, 0.4)', animation: 'sonar-ping 2s ease-out infinite' }} />
                                <span style={{ position: 'relative', width: '6px', height: '6px', borderRadius: '50%', background: '#b91c1c', boxShadow: '0 0 6px rgba(185, 28, 28, 0.6)' }} />
                            </span>
                            <svg width="28" height="12" viewBox="0 0 28 12" style={{ overflow: 'visible', marginLeft: '-2px' }}>
                                <path d="M0,6 L6,6 L8,2 L10,10 L12,4 L14,8 L16,6 L28,6" fill="none" stroke="#dc2626" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" style={{ strokeDasharray: '30', strokeDashoffset: '0', animation: 'ecg-draw 2s linear infinite' }} />
                            </svg>
                            System Status
                        </a>
                    </div>

                    {/* Title & Content */}
                    <div style={{ flex: isMobile ? 'none' : 1, display: 'flex', flexDirection: 'column', justifyContent: isMobile ? 'flex-start' : 'center' }}>
                        <h1 style={{
                            fontSize: isSmallMobile ? '1.5rem' : isMobile ? '1.75rem' : '2rem',
                            fontWeight: 700, lineHeight: 1.1,
                            marginBottom: '0.8rem', letterSpacing: '-0.8px', color: '#e4e5eb'
                        }}>
                            Advanced RAG <br />
                            <span style={{
                                background: 'linear-gradient(135deg, #10b981 0%, #34d399 100%)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent'
                            }}>Recursive Retrieval</span>
                        </h1>
                        <p style={{
                            color: '#8b8fa3',
                            fontSize: isSmallMobile ? '0.8rem' : '0.9rem',
                            marginBottom: isMobile ? '1.5rem' : '2rem',
                            lineHeight: 1.5
                        }}>
                            Production-grade RAG pipeline with Parent-Child chunking,
                            Hybrid Vector Search, PII Masking &amp; LangGraph orchestration.
                        </p>

                        {/* Google Button */}
                        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: isMobile ? '1.5rem' : '2.5rem' }}>
                            <a href={getLoginUrl()} 
                                style={{
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    gap: '0.8rem',
                                    padding: isSmallMobile ? '0.7rem 1.2rem' : '0.75rem 1.8rem',
                                    background: '#ffffff', color: '#3c4043', fontWeight: 500, borderRadius: '8px',
                                    textDecoration: 'none',
                                    fontSize: '0.9rem', fontFamily: '"Google Sans", Roboto, Arial, sans-serif',
                                    boxShadow: '0 1px 3px rgba(60,64,67,0.15)', border: '1px solid #dadce0',
                                    flex: 1,
                                    transition: 'all 0.2s ease-in-out'
                                }}
                                onMouseOver={(e) => {
                                    e.currentTarget.style.background = '#f8f9fa';
                                    e.currentTarget.style.boxShadow = '0 1px 3px rgba(60,64,67,0.3)';
                                }}
                                onMouseOut={(e) => {
                                    e.currentTarget.style.background = '#ffffff';
                                    e.currentTarget.style.boxShadow = '0 1px 3px rgba(60,64,67,0.15)';
                                }}
                            >
                                <svg viewBox="0 0 24 24" width="18" height="18">
                                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" />
                                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                                </svg>
                                Sign in with Google
                            </a>

                            <a href="https://ambuj-ai-portfolio.vercel.app/" target="_blank" rel="noopener noreferrer" 
                                style={{
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    gap: '0.8rem',
                                    padding: isSmallMobile ? '0.7rem 1.2rem' : '0.75rem 1.8rem',
                                    background: 'rgba(255,255,255,0.05)', color: '#fff', fontWeight: 500, borderRadius: '8px',
                                    textDecoration: 'none',
                                    fontSize: '0.9rem',
                                    boxShadow: '0 1px 4px rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)',
                                    flex: 1,
                                    transition: 'all 0.3s ease',
                                    backdropFilter: 'blur(10px)'
                                }}
                                onMouseOver={(e) => {
                                    e.currentTarget.style.background = 'rgba(255,255,255,0.1)';
                                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)';
                                    e.currentTarget.style.boxShadow = '0 4px 15px rgba(255,255,255,0.1)';
                                    e.currentTarget.style.transform = 'translateY(-1px)';
                                }}
                                onMouseOut={(e) => {
                                    e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)';
                                    e.currentTarget.style.boxShadow = '0 1px 4px rgba(0,0,0,0.2)';
                                    e.currentTarget.style.transform = 'translateY(0)';
                                }}
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <circle cx="12" cy="12" r="10"></circle>
                                    <line x1="2" y1="12" x2="22" y2="12"></line>
                                    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                                </svg>
                                View Portfolio
                            </a>
                        </div>

                        {/* Creator */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <img
                                src="/branding/qr.png"
                                alt="Portfolio QR"
                                style={{
                                    width: isSmallMobile ? '60px' : '80px',
                                    height: isSmallMobile ? '60px' : '80px',
                                    borderRadius: '10px',
                                    border: '1px solid rgba(255,255,255,0.1)'
                                }}
                                onError={(e) => { e.target.style.display = 'none' }}
                            />
                            <div>
                                <p style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '1.5px', color: '#5a5e72', marginBottom: '0.2rem' }}>Created by</p>
                                <h3 style={{ fontWeight: 600, fontSize: isSmallMobile ? '0.85rem' : '0.95rem', color: '#ededed' }}>Ambuj Kumar Tripathi</h3>
                                <p style={{ fontSize: '0.75rem', color: '#10b981' }}>AI Engineer &amp; RAG Specialist</p>
                            </div>
                        </div>
                    </div>

                    <div style={{ marginTop: '24px', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '24px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                            <span style={{ fontSize: '18px' }}>🐦</span>
                            <h3 style={{ color: '#fff', fontSize: '14px', fontWeight: '600', letterSpacing: '1px' }}>Recognized by Hugging Face</h3>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
                            <blockquote className="twitter-tweet" data-theme="dark">
                                <p lang="en" dir="ltr">Meet Ambuj-Tripathi-Indian-Legal-Llama-GGUF: a specialized AI model fine-tuned for Indian law. This isn&#39;t just another chatbot. It&#39;s a legal assistant trained to understand the nuances of Indian statutes, case law, and legal language. A game-changer for legal tech in India. <a href="https://t.co/SkLzeaDgpE">pic.twitter.com/SkLzeaDgpE</a></p>&mdash; Hugging Models (@HuggingModels) <a href="https://x.com/HuggingModels/status/2044027666324697451?ref_src=twsrc%5Etfw">April 14, 2026</a>
                            </blockquote>
                        </div>
                    </div>
                </div>

                {/* ============ RIGHT SIDE — Architecture ============ */}
                <div style={{
                    flex: 1,
                    background: '#050505',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    position: 'relative',
                    overflow: 'hidden',
                    minHeight: isMobile ? '300px' : 'auto',
                    padding: isMobile ? '1.5rem' : 0
                }}>
                    {/* Grid pattern background */}
                    <div style={{
                        position: 'absolute',
                        top: 0, left: 0, right: 0, bottom: 0,
                        backgroundImage: 'linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)',
                        backgroundSize: '30px 30px',
                        opacity: 0.4,
                        pointerEvents: 'none',
                        zIndex: 1
                    }} />

                    {/* Architecture Animated SVG */}
                    <object
                        type="image/svg+xml"
                        data="/branding/architecture_animated.svg"
                        aria-label="RAG Architecture"
                        style={{
                            position: 'relative',
                            zIndex: 2,
                            width: isMobile ? '95%' : '90%',
                            maxHeight: isMobile ? '280px' : '90%',
                            borderRadius: '12px',
                            filter: 'drop-shadow(0 8px 40px rgba(0,0,0,0.6))',
                            pointerEvents: 'none'
                        }}
                    >
                        Architecture Diagram
                    </object>
                </div>
            </div>

            {/* ============ BOTTOM — Full-width Tech Stack Marquee ============ */}
            <div style={{
                width: '100%',
                borderTop: '1px solid rgba(255,255,255,0.06)',
                background: '#0a0a0a',
                padding: isSmallMobile ? '0.6rem 1rem' : '0.8rem 2rem',
                display: 'flex',
                alignItems: 'center',
                gap: isSmallMobile ? '0.75rem' : '1.5rem',
                zIndex: 5,
                flexShrink: 0
            }}>
                <p style={{ fontSize: '0.6rem', textTransform: 'uppercase', letterSpacing: '2px', color: '#5a5e72', whiteSpace: 'nowrap', flexShrink: 0 }}>TECH STACK</p>
                <div style={{ flex: 1, overflow: 'hidden', maskImage: 'linear-gradient(90deg, transparent, black 3%, black 97%, transparent)' }}>
                    <div style={{ display: 'flex', animation: 'marquee 20s linear infinite', width: 'max-content' }}>
                        {[0, 1].map((loop) => (
                            <div key={loop} style={{ display: 'flex', gap: isSmallMobile ? '1rem' : '2rem', paddingRight: isSmallMobile ? '1rem' : '2rem' }}>
                                {techStack.map((tech, i) => (
                                    <span key={`${loop}-${i}`} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: isSmallMobile ? '0.7rem' : '0.8rem', whiteSpace: 'nowrap' }} title={tech.name}>
                                        <span style={{ color: tech.color, fontSize: isSmallMobile ? '0.85rem' : '1rem' }}>{tech.icon}</span>
                                        <span style={{ color: tech.color, fontWeight: 500 }}>{tech.name}</span>
                                    </span>
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
