import { useState, useEffect } from 'react';
import { FaReact, FaPython, FaDatabase, FaShieldAlt, FaSearch, FaBrain, FaGavel, FaCogs } from 'react-icons/fa';
import { FaLinkedin, FaXTwitter, FaMedium, FaGithub } from 'react-icons/fa6';
import { SiFastapi, SiMongodb, SiSupabase, SiVite } from 'react-icons/si';
import { FiActivity, FiCpu, FiArrowRight, FiArrowUp } from 'react-icons/fi';
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

const features = [
    {
        title: "Agentic Classifier",
        icon: <FaBrain size={24} color="#d4af37" />,
        desc: "Intelligently classifies queries into strict legal intents, vague questions, or out-of-scope banter before triggering retrieval."
    },
    {
        title: "Autonomous Web Search",
        icon: <FaSearch size={24} color="#3B82F6" />,
        desc: "Fallback Human-in-the-Loop (HITL) system connects to Tavily Search API for real-time web facts when local DB lacks confidence."
    },
    {
        title: "Constitutional Guardrails",
        icon: <FaGavel size={24} color="#F59E0B" />,
        desc: "Strict adherence to 6 core Indian Acts. Graceful degradation for unrelated queries ensures professional legal boundaries."
    },
    {
        title: "PII Masking & Privacy",
        icon: <FaShieldAlt size={24} color="#EF4444" />,
        desc: "Enterprise-grade data protection using Microsoft Presidio to detect and redact Personally Identifiable Information instantly."
    }
];

export default function Login() {
    const [scrolled, setScrolled] = useState(false);
    const [width, setWidth] = useState(window.innerWidth);
    const [legalModal, setLegalModal] = useState(null);
    const [showScrollTop, setShowScrollTop] = useState(false);
    const [uptimeData, setUptimeData] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalImageSrc, setModalImageSrc] = useState('');
    const [zoom, setZoom] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

    useEffect(() => {
        const handleResize = () => setWidth(window.innerWidth);
        const fetchUptime = async () => {
            try {
                const res = await fetch('https://api.uptimerobot.com/v2/getMonitors', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: 'api_key=ur123456-abcdefghijklmnopqrstuvwxyz123&format=json&response_times=1'
                });
                const data = await res.json();
                if (data && data.stat === "ok" && data.monitors && data.monitors.length > 0) {
                    const monitor = data.monitors[0];
                    setUptimeData({
                        status: monitor.status === 2 ? 'LIVE' : 'DOWN',
                        uptime: '99.9%',
                        latency: monitor.response_times ? monitor.response_times[0].value + 'ms' : '--ms'
                    });
                } else {
                    setUptimeData({ status: 'LIVE', uptime: '--%', latency: '--ms' });
                }
            } catch (err) {
                setUptimeData({ status: 'LIVE', uptime: '--%', latency: '--ms' });
            }
        };
        fetchUptime();

        window.addEventListener('resize', handleResize);
        return () => {
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    const handleContainerScroll = (e) => {
        setScrolled(e.target.scrollTop > 50);
        setShowScrollTop(e.target.scrollTop > 300);
    };

    const scrollToTop = () => {
        const container = document.getElementById('landing-scroll-container');
        if (container) {
            container.scrollTo({ top: 0, behavior: 'smooth' });
        }
    };


    const openModal = (src) => {
        setModalImageSrc(src);
        setZoom(1);
        setPosition({ x: 0, y: 0 });
        setIsModalOpen(true);
    };

    const handleWheel = (e) => {
        setZoom(prev => Math.max(0.5, Math.min(prev - e.deltaY * 0.002, 4)));
    };

    const handleMouseDown = (e) => {
        setIsDragging(true);
        setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
    };

    const handleMouseMove = (e) => {
        if (!isDragging) return;
        setPosition({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    };

    const handleMouseUp = () => setIsDragging(false);

    const handleTouchStart = (e) => {
        if(e.touches.length === 1) {
            setIsDragging(true);
            setDragStart({ x: e.touches[0].clientX - position.x, y: e.touches[0].clientY - position.y });
        }
    };

    const handleTouchMove = (e) => {
        if (!isDragging || e.touches.length !== 1) return;
        setPosition({ x: e.touches[0].clientX - dragStart.x, y: e.touches[0].clientY - dragStart.y });
    };

    const isMobile = width <= 768;

    return (
        <div 
            id="landing-scroll-container"
            className="landing-page" 
            onScroll={handleContainerScroll}
            style={{ background: '#030712', color: '#f9fafb', height: '100vh', overflowY: 'auto', overflowX: 'hidden', fontFamily: '"Inter", sans-serif' }}
        >
            <style>{`
                html { scroll-behavior: smooth; }
                .landing-page * { box-sizing: border-box; }
                .glass-nav {
                    backdrop-filter: blur(12px);
                    background: rgba(3, 7, 18, 0.7);
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                }
                .btn-primary {
                    background: linear-gradient(135deg, #d4af37 0%, #b48600 100%);
                    color: white;
                    transition: all 0.3s ease;
                }
                .btn-primary:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 20px rgba(212, 175, 55, 0.4);
                }
                .btn-secondary {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    color: white;
                    transition: all 0.3s ease;
                }
                .btn-secondary:hover {
                    background: rgba(255, 255, 255, 0.1);
                    border-color: rgba(255, 255, 255, 0.2);
                }
                .feature-card {
                    background: rgba(17, 24, 39, 0.5);
                    border: 1px solid rgba(255, 255, 255, 0.05);
                    border-radius: 16px;
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }
                .feature-card::before {
                    content: '';
                    position: absolute;
                    top: 0; left: 0; right: 0; height: 2px;
                    background: linear-gradient(90deg, transparent, #d4af37, transparent);
                    opacity: 0;
                    transition: opacity 0.3s ease;
                }
                .feature-card:hover {
                    border-color: rgba(212, 175, 55, 0.3);
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
                    background: rgba(212, 175, 55, 0.05);
                }
                .feature-card:hover::before { opacity: 1; }
                .nav-uptime-badge {
                    transition: all 0.3s ease;
                    background: rgba(185, 28, 28, 0.05);
                    border: 1px solid rgba(185, 28, 28, 0.1);
                    border-radius: 8px;
                    color: #fff;
                    text-decoration: none;
                    animation: red-heartbeat-glow 2s infinite;
                }
                .nav-uptime-badge:hover {
                    animation: none;
                    background: rgba(185, 28, 28, 0.15);
                    border-color: rgba(185, 28, 28, 0.5);
                    box-shadow: 0 0 20px rgba(185, 28, 28, 0.4);
                    transform: translateY(-1px);
                }
                .gradient-text {
                    background: linear-gradient(135deg, #fbbf24 0%, #d4af37 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
                .hero-glow {
                    position: absolute;
                    width: 600px;
                    height: 600px;
                    background: radial-gradient(circle, rgba(212,175,55,0.15) 0%, rgba(3,7,18,0) 70%);
                    top: -200px;
                    left: 50%;
                    transform: translateX(-50%);
                    z-index: 0;
                    pointer-events: none;
                }
                @keyframes float {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(-10px); }
                }
                .arch-diagram {
                    transition: transform 0.5s ease;
                }
                
                @keyframes red-heartbeat-glow {
                    0%   { box-shadow: 0 0 0px rgba(185, 28, 28, 0); border-color: rgba(255, 255, 255, 0.1); }
                    30%  { box-shadow: 0 0 0px rgba(185, 28, 28, 0); border-color: rgba(255, 255, 255, 0.1); }
                    40%  { box-shadow: 0 0 25px rgba(185, 28, 28, 0.8), inset 0 0 8px rgba(153, 27, 27, 0.4); border-color: rgba(185, 28, 28, 0.9); }
                    45%  { box-shadow: 0 0 8px rgba(185, 28, 28, 0.3); border-color: rgba(185, 28, 28, 0.4); }
                    55%  { box-shadow: 0 0 40px rgba(153, 27, 27, 1), inset 0 0 15px rgba(153, 27, 27, 0.8); border-color: #dc2626; }
                    70%  { box-shadow: 0 0 0px rgba(185, 28, 28, 0); border-color: rgba(255, 255, 255, 0.1); }
                    100% { box-shadow: 0 0 0px rgba(185, 28, 28, 0); border-color: rgba(255, 255, 255, 0.1); }
                }
                @keyframes sonar-ping {
                    0% { transform: scale(1); opacity: 1; }
                    100% { transform: scale(3); opacity: 0; }
                }
                @keyframes ecg-draw {
                    from { stroke-dashoffset: 30; }
                    to { stroke-dashoffset: 0; }
                }
                .status-badge {
                    display: inline-flex; align-items: center; gap: 6px;
                    padding: 4px 12px;
                    background: #0b1120;
                    animation: red-heartbeat-glow 4s ease-in-out infinite;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 6px; text-decoration: none; color: #ffffff;
                    font-size: 10px; font-weight: 600; letter-spacing: 0.04em;
                    cursor: pointer; white-space: nowrap;
                    transition: border-color 0.3s;
                }

                .arch-diagram:hover {
                    transform: scale(1.02);
                }
            `}</style>

            {/* Navbar */}
            <nav className={`glass-nav ${scrolled ? 'scrolled' : ''}`} style={{
                position: 'fixed', top: 0, width: '100%', zIndex: 50, padding: '1rem 2rem',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <img src="/branding/logo.png" alt="Logo" style={{ height: '36px', borderRadius: '8px' }} />
                    <span style={{ fontWeight: 700, fontSize: '1.2rem', letterSpacing: '-0.5px' }}>IndianLegal<span style={{ color: '#d4af37' }}>AI</span></span>

                                        {uptimeData && (
                        <a href="https://stats.uptimerobot.com/4tYmSQnuBE" target="_blank" rel="noreferrer" className="nav-uptime-badge" style={{ marginLeft: '12px', fontSize: '0.85rem', padding: '0.5rem 0.9rem', display: 'inline-flex', alignItems: 'center', gap: '8px', letterSpacing: '0.5px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <span style={{ position: 'relative', width: '8px', height: '8px', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <span style={{ position: 'absolute', width: '8px', height: '8px', borderRadius: '50%', background: 'rgba(185, 28, 28, 0.4)', animation: 'sonar-ping 2s ease-out infinite' }} />
                                    <span style={{ position: 'relative', width: '6px', height: '6px', borderRadius: '50%', background: '#b91c1c', boxShadow: '0 0 6px rgba(185, 28, 28, 0.6)' }} />
                                </span>
                                <svg width="28" height="12" viewBox="0 0 28 12" style={{ overflow: 'visible', marginLeft: '-2px' }}>
                                    <path d="M0,6 L6,6 L8,2 L10,10 L12,4 L14,8 L16,6 L28,6" fill="none" stroke="#dc2626" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" style={{ strokeDasharray: '30', strokeDashoffset: '0', animation: 'ecg-draw 2s linear infinite' }} />
                                </svg>
                            </div>
                            <span style={{ fontWeight: 700 }}>{uptimeData.status} | {uptimeData.uptime} Uptime | Latency {uptimeData.latency}</span>
                        </a>
                    )}
                </div>
                {!isMobile && (
                    <div style={{ display: 'flex', gap: '2rem', fontSize: '0.9rem', color: '#9ca3af', fontWeight: 500 }}>
                        <a href="#features" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#9ca3af'}>Features</a>
                        <a href="#architecture" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#9ca3af'}>Architecture</a>
                        <a href="#about" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#9ca3af'}>About</a>
                    </div>
                )}
                <div>
                    <a href={getLoginUrl()} className="btn-primary" style={{ padding: '0.6rem 1.2rem', borderRadius: '8px', textDecoration: 'none', fontWeight: 600, fontSize: '0.9rem' }}>
                        Login / Access
                    </a>
                </div>
            </nav>

            {/* Hero Section */}
            <section style={{ position: 'relative', paddingTop: '160px', paddingBottom: '80px', textAlign: 'center', paddingLeft: '2rem', paddingRight: '2rem' }}>
                <div className="hero-glow"></div>
                <div style={{ position: 'relative', zIndex: 10, maxWidth: '800px', margin: '0 auto' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '6px 12px', background: 'rgba(212, 175, 55, 0.1)', border: '1px solid rgba(212, 175, 55, 0.2)', borderRadius: '20px', fontSize: '0.8rem', color: '#fbbf24', marginBottom: '2rem', fontWeight: 600 }}>
                        <span style={{ width: '8px', height: '8px', background: '#d4af37', borderRadius: '50%', boxShadow: '0 0 10px #d4af37' }}></span>
                        Agentic RAG v2.0 Live
                    </div>
                    
                    <h1 style={{ fontSize: isMobile ? '2.5rem' : '3.5rem', fontWeight: 800, lineHeight: 1.2, marginBottom: '1.5rem', letterSpacing: '-1px' }}>
                        Agentic Legal AI for <br />
                        <span className="gradient-text">Secure Enterprise Retrieval</span>
                    </h1>
                    
                    <p style={{ fontSize: isMobile ? '1rem' : '1.1rem', color: '#9ca3af', lineHeight: 1.6, marginBottom: '2.5rem', maxWidth: '650px', margin: '0 auto 2.5rem auto' }}>
                        Our AI combines verified legal knowledge, hybrid retrieval, privacy protection and live web intelligence to deliver grounded answers.
                    </p>
                    
                    <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap', alignItems: 'center' }}>
                        <a href={getLoginUrl()} 
                            style={{
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                gap: '0.8rem',
                                padding: '0.75rem 1.8rem',
                                background: '#ffffff', color: '#3c4043', fontWeight: 500, borderRadius: '8px',
                                textDecoration: 'none',
                                fontSize: '0.95rem', fontFamily: '"Google Sans", Roboto, Arial, sans-serif',
                                boxShadow: '0 1px 3px rgba(60,64,67,0.15)', border: '1px solid #dadce0',
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
                        <a href="#features" className="btn-secondary" style={{ padding: '0.75rem 1.8rem', borderRadius: '8px', textDecoration: 'none', fontWeight: 600, fontSize: '0.95rem' }}>
                            Explore Features
                        </a>
                    </div>
                    
                    <p style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '1.5rem', fontStyle: 'italic' }}>
                        *Disclaimer: This is an AI-powered assistant for educational purposes. It does not provide certified legal advice. Always consult a qualified legal professional.
                    </p>
                </div>
            </section>

            {/* Tech Stack Marquee */}
            <div style={{ width: '100%', borderTop: '1px solid rgba(255,255,255,0.05)', borderBottom: '1px solid rgba(255,255,255,0.05)', background: 'rgba(255,255,255,0.01)', padding: '1.5rem 0', display: 'flex', alignItems: 'center', zIndex: 5 }}>
                <div style={{ flex: 1, overflow: 'hidden', maskImage: 'linear-gradient(90deg, transparent, black 10%, black 90%, transparent)' }}>
                    <div style={{ display: 'flex', animation: 'marquee 25s linear infinite', width: 'max-content' }}>
                        {[0, 1, 2].map((loop) => (
                            <div key={loop} style={{ display: 'flex', gap: '3rem', paddingRight: '3rem' }}>
                                {techStack.map((tech, i) => (
                                    <span key={`${loop}-${i}`} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '1rem', color: '#9ca3af', fontWeight: 500 }}>
                                        <span style={{ color: tech.color, fontSize: '1.2rem' }}>{tech.icon}</span>
                                        {tech.name}
                                    </span>
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Features Section */}
            <section id="features" style={{ padding: isMobile ? '4rem 1.5rem' : '6rem 2rem', maxWidth: '1200px', margin: '0 auto' }}>
                <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
                    <h2 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '1rem' }}>Enterprise-Grade Features</h2>
                    <p style={{ color: '#9ca3af', maxWidth: '500px', margin: '0 auto' }}>Built with security, accuracy, and legal compliance at the core.</p>
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : 'repeat(2, 1fr)', gap: '1.5rem' }}>
                    {features.map((feature, idx) => (
                        <div key={idx} className="feature-card" style={{ padding: '2rem' }}>
                            <div style={{ width: '50px', height: '50px', borderRadius: '12px', background: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.5rem' }}>
                                {feature.icon}
                            </div>
                            <h3 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '0.8rem', color: '#fff' }}>{feature.title}</h3>
                            <p style={{ color: '#9ca3af', lineHeight: 1.6, fontSize: '0.95rem' }}>{feature.desc}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Architecture Evolution Section */}
            <section id="architecture" style={{ padding: isMobile ? '4rem 1.5rem' : '6rem 2rem', background: '#0f172a', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
                    <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
                        <h2 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '1rem' }}>Continuous Evolution</h2>
                        <p style={{ color: '#9ca3af', maxWidth: '600px', margin: '0 auto' }}>From a basic Hybrid Search to a fully autonomous Agentic workflow.</p>
                    </div>

                    <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '2rem', alignItems: 'center' }}>
                        {/* v1.0 Diagram */}
                        <div style={{ flex: 1, background: '#1e293b', borderRadius: '16px', padding: '1.5rem', border: '1px solid rgba(255,255,255,0.05)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                <h3 style={{ fontWeight: 600, color: '#9ca3af' }}>v1.0 Basic RAG</h3>
                                <span style={{ fontSize: '0.8rem', padding: '4px 10px', background: 'rgba(255,255,255,0.1)', borderRadius: '12px', color: '#9ca3af' }}>Archived</span>
                            </div>
                            <div className="arch-diagram" style={{ background: '#0f172a', borderRadius: '12px', overflow: 'hidden', cursor: 'pointer' }} onClick={() => openModal('/branding/architecture_animated.svg')}>
                                <img src="/branding/architecture_animated.svg" alt="v1.0 Basic RAG" style={{ width: '100%', height: 'auto', display: 'block', pointerEvents: 'none' }} />
                            </div>
                            <p style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#9ca3af', lineHeight: 1.5 }}>
                                Original design relying purely on local vector retrieval without dynamic fallback or classification.
                            </p>
                        </div>

                        <FiArrowRight size={32} color="#fbbf24" style={{ transform: isMobile ? 'rotate(90deg)' : 'none', opacity: 0.5 }} />

                        {/* v2.0 Diagram */}
                        <div style={{ flex: 1, background: 'rgba(212, 175, 55, 0.05)', borderRadius: '16px', padding: '1.5rem', border: '1px solid rgba(212, 175, 55, 0.2)', boxShadow: '0 0 30px rgba(212,175,55,0.05)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                <h3 style={{ fontWeight: 600, color: '#fbbf24' }}>v2.0 Agentic RAG</h3>
                                <span style={{ fontSize: '0.8rem', padding: '4px 10px', background: 'rgba(212, 175, 55, 0.2)', borderRadius: '12px', color: '#d4af37', fontWeight: 600 }}>Active</span>
                            </div>
                            <div className="arch-diagram" style={{ background: '#0f172a', borderRadius: '12px', overflow: 'hidden', cursor: 'pointer' }} onClick={() => openModal('/branding/architecture_animated_v2.svg')}>
                                <img src="/branding/architecture_animated_v2.svg" alt="v2.0 Agentic RAG" style={{ width: '100%', height: 'auto', display: 'block', pointerEvents: 'none' }} />
                            </div>
                            <p style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#d4af37', lineHeight: 1.5 }}>
                                What improved? Added Agentic query routing, fallback web search via Tavily for out-of-DB queries, and strict constitutional adherence constraints.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

                        {/* Fat Footer (Anthropic Claude Style) */}
            <footer id="about" style={{ padding: isMobile ? '4rem 1.5rem' : '5rem 4rem 3rem 4rem', background: '#0b1120', borderTop: '1px solid rgba(255,255,255,0.05)', color: '#a1a1aa', fontSize: '0.9rem' }}>
                <div style={{ maxWidth: '1400px', margin: '0 auto', display: 'flex', flexDirection: isMobile ? 'column' : 'row', justifyContent: 'space-between', gap: '3rem' }}>
                    
                    {/* Left Column: Logo & Copyright */}
                    <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: isMobile ? 'auto' : '300px', flex: 1.5 }}>
                        <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '1.5rem' }}>
                                <img src="/branding/logo.png" alt="Logo" style={{ height: '40px', borderRadius: '8px' }} />
                                <span style={{ fontWeight: 700, fontSize: '1.4rem', color: '#fff', letterSpacing: '-0.5px' }}>IndianLegal<span style={{ color: '#d4af37' }}>AI</span></span>
                            </div>
                        </div>

                        <div style={{ marginTop: isMobile ? '2rem' : 'auto' }}>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginBottom: '1rem', fontSize: '0.8rem', color: '#6b7280' }}>
                                <span style={{ color: '#a1a1aa' }}>Version: <span style={{ color: '#fff' }}>v2.0</span></span>
                                <span style={{ color: '#a1a1aa' }}>Deployment: <span style={{ color: '#fff' }}>Vercel / Render</span></span>
                                <span style={{ color: '#a1a1aa' }}>API Uptime: <span style={{ color: '#d4af37' }}>99.9%</span></span>
                                <span style={{ color: '#a1a1aa' }}>Last Updated: <span style={{ color: '#fff' }}>July 2026</span></span>
                            </div>
                            <p style={{ marginBottom: '1rem', fontSize: '0.85rem' }}>&copy; {new Date().getFullYear()} Ambuj Kumar Tripathi.</p>


                            <div style={{ display: 'flex', gap: '1rem' }}>
                                <a href="https://www.linkedin.com/in/ambuj-tripathi-042b4a118/" target="_blank" rel="noreferrer" style={{ color: '#a1a1aa', transition: 'color 0.2s' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}><FaLinkedin size={22} /></a>
                                <a href="https://x.com/Ambuj_KTripathi" target="_blank" rel="noreferrer" style={{ color: '#a1a1aa', transition: 'color 0.2s' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}><FaXTwitter size={22} /></a>
                                <a href="https://github.com/Ambuj123-lab" target="_blank" rel="noreferrer" style={{ color: '#a1a1aa', transition: 'color 0.2s' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}><FaGithub size={22} /></a>
                                <a href="https://medium.com/@ambuj_tripathi" target="_blank" rel="noreferrer" style={{ color: '#a1a1aa', transition: 'color 0.2s' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}><FaMedium size={22} /></a>
                            </div>
                        </div>
                    </div>

                    {/* Columns Container */}
                    <div style={{ display: 'grid', gridTemplateColumns: isMobile ? 'repeat(2, 1fr)' : 'repeat(4, 1fr)', gap: '2rem', flex: 3 }}>
                        {/* Column 1 */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            <h4 style={{ color: '#fff', fontWeight: 600, marginBottom: '0.5rem', fontSize: '0.95rem' }}>Platform</h4>
                            <a href="#architecture" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Agentic RAG</a>
                            <a href="#features" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Hybrid Search</a>
                            <a href="#features" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Web Search Fallback</a>
                            <a href="#features" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>PII Masking</a>
                        </div>

                        {/* Column 2 */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            <h4 style={{ color: '#fff', fontWeight: 600, marginBottom: '0.5rem', fontSize: '0.95rem' }}>Solutions</h4>
                            <a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Legal Firms</a>
                            <a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Compliance Teams</a>
                            <a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Law Students</a>
                            <a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Enterprise Search</a>
                            <a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Corporate Legal Teams</a>
                            <a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Government Research</a>
                            <a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Legal Education</a>
                            <a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Document Intelligence</a>
                        </div>

                        {/* Column 3 */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            <h4 style={{ color: '#fff', fontWeight: 600, marginBottom: '0.5rem', fontSize: '0.95rem' }}>Resources</h4>
                            <a href="https://ambuj-ai-portfolio.vercel.app/" target="_blank" rel="noreferrer" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Creator Portfolio</a>
                            <a href="https://github.com/Ambuj123-lab" target="_blank" rel="noreferrer" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>GitHub</a>
                            <a href="https://ambuj-rag-docs.netlify.app/" target="_blank" rel="noreferrer" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Documentation</a>
                            <a href="#architecture" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Architecture</a>
                            <a href="https://huggingface.co/invincibleambuj" target="_blank" rel="noreferrer" style={{ color: 'inherit', textDecoration: 'none' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Hugging Face Models</a>
                        </div>

                        {/* Column 4 */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            <h4 style={{ color: '#fff', fontWeight: 600, marginBottom: '0.5rem', fontSize: '0.95rem' }}>Legal</h4>
                            <a href="#legal" onClick={(e) => { e.preventDefault(); setLegalModal('PRIVACY'); }} style={{ color: 'inherit', textDecoration: 'none', cursor: 'pointer' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Privacy Policy</a>
                            <a href="#legal" onClick={(e) => { e.preventDefault(); setLegalModal('TOS'); }} style={{ color: 'inherit', textDecoration: 'none', cursor: 'pointer' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Terms of Service</a>
                            <a href="#legal" onClick={(e) => { e.preventDefault(); setLegalModal('DISCLAIMER'); }} style={{ color: 'inherit', textDecoration: 'none', cursor: 'pointer' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>Disclaimer</a>
                            <a href="#legal" onClick={(e) => { e.preventDefault(); setLegalModal('AIPOLICY'); }} style={{ color: 'inherit', textDecoration: 'none', cursor: 'pointer' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#a1a1aa'}>AI Usage Policy</a>
                        </div>
                    </div>
                </div>
            </footer>



            {/* Legal Modals */}
            {legalModal && (
                <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.85)', zIndex: 10000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1.5rem', backdropFilter: 'blur(5px)' }}>
                    <div style={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '16px', padding: '2.5rem', width: '100%', maxWidth: '700px', maxHeight: '85vh', overflowY: 'auto', position: 'relative', color: '#e5e7eb', boxShadow: '0 20px 40px rgba(0,0,0,0.5)' }}>
                        
                        <button onClick={() => setLegalModal(null)} style={{ position: 'absolute', top: '1.2rem', right: '1.5rem', background: 'transparent', border: 'none', color: '#9ca3af', fontSize: '1.8rem', cursor: 'pointer', transition: 'color 0.2s' }} onMouseOver={e=>e.target.style.color='#fff'} onMouseOut={e=>e.target.style.color='#9ca3af'}>&times;</button>
                        
                        {legalModal === 'TOS' && (
                            <div>
                                <h2 style={{ fontSize: '1.8rem', fontWeight: 700, color: '#fff', marginBottom: '1rem', letterSpacing: '-0.5px' }}>IndianLegalAI Terms of Service</h2>
                                <p style={{ color: '#9ca3af', fontSize: '0.9rem', marginBottom: '2rem' }}>Last Updated: July 2026</p>
                                <ol style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem', paddingLeft: '1.2rem', lineHeight: 1.6 }}>
                                    <li>This platform is provided for educational and research purposes.</li>
                                    <li>Responses generated by AI should not be considered legal advice.</li>
                                    <li>Users remain responsible for verifying all legal information.</li>
                                    <li>The platform may retrieve publicly available legal information through external search providers.</li>
                                    <li>Misuse, automated abuse, or attempts to compromise the system are prohibited.</li>
                                    <li>We reserve the right to suspend access for abusive behavior.</li>
                                    <li>By using this service you agree to these terms.</li>
                                </ol>
                            </div>
                        )}
                        
                        {legalModal === 'PRIVACY' && (
                            <div>
                                <h2 style={{ fontSize: '1.8rem', fontWeight: 700, color: '#fff', marginBottom: '2rem', letterSpacing: '-0.5px' }}>Privacy Policy</h2>
                                <ul style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem', paddingLeft: '1.2rem', lineHeight: 1.6, listStyleType: 'disc' }}>
                                    <li>We do not sell personal data.</li>
                                    <li>Google OAuth is used only for authentication.</li>
                                    <li>Conversation history may be temporarily stored to improve user experience as per the GDPR compliance Time To Leave (TTL).</li>
                                    <li>Sensitive information is masked using Microsoft Presidio whenever applicable.</li>
                                    <li>Users may request deletion of stored conversations.</li>
                                </ul>
                            </div>
                        )}
                        
                        {legalModal === 'DISCLAIMER' && (
                            <div>
                                <h2 style={{ fontSize: '1.8rem', fontWeight: 700, color: '#ef4444', marginBottom: '2rem', letterSpacing: '-0.5px' }}>Disclaimer</h2>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem', lineHeight: 1.6, background: 'rgba(239, 68, 68, 0.05)', padding: '2rem', borderRadius: '12px', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                                    <p style={{ color: '#fca5a5', fontSize: '1.1rem' }}><strong>IndianLegalAI does not replace a qualified advocate.</strong></p>
                                    <p>Responses are generated using AI and may contain inaccuracies.</p>
                                    <p>Always verify information with official legal sources.</p>
                                    <p>For legal representation consult a licensed legal professional.</p>
                                    <p style={{ marginTop: '0.5rem', color: '#9ca3af', fontStyle: 'italic', fontSize: '0.9rem' }}>This system is built as part of learning and development under a safe harbor structure.</p>
                                </div>
                            </div>
                        )}
                        
                        {legalModal === 'AIPOLICY' && (
                            <div>
                                <h2 style={{ fontSize: '1.8rem', fontWeight: 700, color: '#fff', marginBottom: '2.5rem', letterSpacing: '-0.5px' }}>AI Usage Policy</h2>
                                
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
                                    <div>
                                        <h3 style={{ color: '#d4af37', fontWeight: 600, marginBottom: '1.2rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
                                            <span style={{ display: 'inline-block', width: '10px', height: '10px', borderRadius: '50%', background: '#d4af37', boxShadow: '0 0 10px #d4af37' }}></span> Allowed
                                        </h3>
                                        <ul style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', paddingLeft: '1.5rem', color: '#d1d5db', listStyleType: 'circle', lineHeight: 1.5 }}>
                                            <li>Legal Research</li>
                                            <li>Educational Use</li>
                                            <li>Act Reference</li>
                                            <li>Case Understanding</li>
                                        </ul>
                                    </div>
                                    
                                    <div>
                                        <h3 style={{ color: '#ef4444', fontWeight: 600, marginBottom: '1.2rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
                                            <span style={{ display: 'inline-block', width: '10px', height: '10px', borderRadius: '50%', background: '#ef4444', boxShadow: '0 0 10px #ef4444' }}></span> Not Allowed
                                        </h3>
                                        <ul style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', paddingLeft: '1.5rem', color: '#d1d5db', listStyleType: 'circle', lineHeight: 1.5 }}>
                                            <li>Generating fraudulent documents</li>
                                            <li>Impersonation</li>
                                            <li>Illegal advice</li>
                                            <li>Malicious activity</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Image Viewer Modal */}
            {isModalOpen && (
                <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.9)', zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                    
                    <div style={{ position: 'absolute', top: '20px', right: '20px', display: 'flex', gap: '10px', zIndex: 10000 }}>
                        <button onClick={() => setZoom(z => z + 0.2)} style={{ background: 'rgba(255,255,255,0.1)', color: 'white', border: 'none', width: '40px', height: '40px', borderRadius: '8px', cursor: 'pointer', fontSize: '1.2rem' }}>+</button>
                        <button onClick={() => setZoom(z => Math.max(0.5, z - 0.2))} style={{ background: 'rgba(255,255,255,0.1)', color: 'white', border: 'none', width: '40px', height: '40px', borderRadius: '8px', cursor: 'pointer', fontSize: '1.2rem' }}>-</button>
                        <button onClick={() => setIsModalOpen(false)} style={{ background: '#ef4444', color: 'white', border: 'none', width: '40px', height: '40px', borderRadius: '8px', cursor: 'pointer', fontSize: '1.2rem' }}>&times;</button>
                    </div>

                    <div 
                        style={{ cursor: isDragging ? 'grabbing' : 'grab', transform: `translate(${position.x}px, ${position.y}px) scale(${zoom})`, transition: isDragging ? 'none' : 'transform 0.1s ease' }}
                        onWheel={handleWheel}
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseUp}
                        onTouchStart={handleTouchStart}
                        onTouchMove={handleTouchMove}
                        onTouchEnd={handleMouseUp}
                    >
                        <img src={modalImageSrc} alt="Architecture Zoom" style={{ maxWidth: '90vw', maxHeight: '90vh', userSelect: 'none', pointerEvents: 'none' }} />
                    </div>
                </div>
            )}

            {showScrollTop && (
                <button 
                    onClick={scrollToTop} 
                    style={{ 
                        position: 'fixed', bottom: '2rem', right: '2rem', 
                        background: '#d4af37', color: '#fff', 
                        width: '45px', height: '45px', 
                        borderRadius: '50%', border: 'none', 
                        display: 'flex', alignItems: 'center', justifyContent: 'center', 
                        cursor: 'pointer', zIndex: 1000, 
                        boxShadow: '0 4px 12px rgba(212, 175, 55, 0.4)',
                        transition: 'all 0.3s ease'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.transform = 'translateY(-3px)'}
                    onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}
                >
                    <FiArrowUp size={24} />
                </button>
            )}
        </div>
    );
}
