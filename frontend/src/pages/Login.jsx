import { useState, useEffect } from 'react';
import { FaReact, FaPython, FaDatabase, FaShieldAlt, FaSearch, FaBrain, FaGavel, FaGithub } from 'react-icons/fa';
import { SiFastapi, SiMongodb, SiSupabase, SiVite } from 'react-icons/si';
import { FiActivity, FiCpu, FiArrowRight } from 'react-icons/fi';
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
        icon: <FaBrain size={24} color="#10B981" />,
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

    useEffect(() => {
        const handleScroll = () => setScrolled(window.scrollY > 50);
        const handleResize = () => setWidth(window.innerWidth);
        window.addEventListener('scroll', handleScroll);
        window.addEventListener('resize', handleResize);
        return () => {
            window.removeEventListener('scroll', handleScroll);
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    const isMobile = width <= 768;

    return (
        <div className="landing-page" style={{ background: '#030712', color: '#f9fafb', minHeight: '100vh', overflowX: 'hidden', fontFamily: '"Inter", sans-serif' }}>
            <style>{`
                html { scroll-behavior: smooth; }
                .landing-page * { box-sizing: border-box; }
                .glass-nav {
                    backdrop-filter: blur(12px);
                    background: rgba(3, 7, 18, 0.7);
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                }
                .btn-primary {
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white;
                    transition: all 0.3s ease;
                }
                .btn-primary:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
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
                    background: linear-gradient(90deg, transparent, #10b981, transparent);
                    opacity: 0;
                    transition: opacity 0.3s ease;
                }
                .feature-card:hover {
                    transform: translateY(-5px);
                    border-color: rgba(16, 185, 129, 0.3);
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                }
                .feature-card:hover::before { opacity: 1; }
                .gradient-text {
                    background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
                .hero-glow {
                    position: absolute;
                    width: 600px;
                    height: 600px;
                    background: radial-gradient(circle, rgba(16,185,129,0.15) 0%, rgba(3,7,18,0) 70%);
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
                    <span style={{ fontWeight: 700, fontSize: '1.2rem', letterSpacing: '-0.5px' }}>IndianLegal<span style={{ color: '#10b981' }}>AI</span></span>
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
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '6px 12px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '20px', fontSize: '0.8rem', color: '#34d399', marginBottom: '2rem', fontWeight: 600 }}>
                        <span style={{ width: '8px', height: '8px', background: '#10b981', borderRadius: '50%', boxShadow: '0 0 10px #10b981' }}></span>
                        Agentic RAG v2.0 Live
                    </div>
                    
                    <h1 style={{ fontSize: isMobile ? '2.5rem' : '4rem', fontWeight: 800, lineHeight: 1.1, marginBottom: '1.5rem', letterSpacing: '-1.5px' }}>
                        Advanced Legal AI <br />
                        <span className="gradient-text">Orchestration Engine</span>
                    </h1>
                    
                    <p style={{ fontSize: isMobile ? '1rem' : '1.2rem', color: '#9ca3af', lineHeight: 1.6, marginBottom: '2.5rem', maxWidth: '600px', margin: '0 auto 2.5rem auto' }}>
                        A production-grade AI pipeline combining Hybrid Vector Search, Microsoft Presidio PII Masking, and Human-in-the-Loop Web Search.
                    </p>
                    
                    <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
                        <a href={getLoginUrl()} className="btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '0.8rem 1.8rem', borderRadius: '8px', textDecoration: 'none', fontWeight: 600, fontSize: '1rem' }}>
                            Sign in with Google <FiArrowRight />
                        </a>
                        <a href="#features" className="btn-secondary" style={{ padding: '0.8rem 1.8rem', borderRadius: '8px', textDecoration: 'none', fontWeight: 600, fontSize: '1rem' }}>
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
            <section id="architecture" style={{ padding: isMobile ? '4rem 1.5rem' : '6rem 2rem', background: '#070b14', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
                    <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
                        <h2 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '1rem' }}>Continuous Evolution</h2>
                        <p style={{ color: '#9ca3af', maxWidth: '600px', margin: '0 auto' }}>From a basic Hybrid Search to a fully autonomous Agentic workflow.</p>
                    </div>

                    <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '2rem', alignItems: 'center' }}>
                        {/* v1.0 Diagram */}
                        <div style={{ flex: 1, background: '#0a0f1c', borderRadius: '16px', padding: '1.5rem', border: '1px solid rgba(255,255,255,0.05)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                <h3 style={{ fontWeight: 600, color: '#9ca3af' }}>v1.0 Basic RAG</h3>
                                <span style={{ fontSize: '0.8rem', padding: '4px 10px', background: 'rgba(255,255,255,0.1)', borderRadius: '12px', color: '#9ca3af' }}>Archived</span>
                            </div>
                            <div className="arch-diagram" style={{ background: '#05080f', borderRadius: '12px', overflow: 'hidden' }}>
                                <object type="image/svg+xml" data="/branding/architecture_animated.svg" style={{ width: '100%', height: 'auto', display: 'block' }}></object>
                            </div>
                        </div>

                        <FiArrowRight size={32} color="#34d399" style={{ transform: isMobile ? 'rotate(90deg)' : 'none', opacity: 0.5 }} />

                        {/* v2.0 Diagram */}
                        <div style={{ flex: 1, background: 'rgba(16, 185, 129, 0.05)', borderRadius: '16px', padding: '1.5rem', border: '1px solid rgba(16, 185, 129, 0.2)', boxShadow: '0 0 30px rgba(16,185,129,0.05)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                <h3 style={{ fontWeight: 600, color: '#34d399' }}>v2.0 Agentic RAG</h3>
                                <span style={{ fontSize: '0.8rem', padding: '4px 10px', background: 'rgba(16, 185, 129, 0.2)', borderRadius: '12px', color: '#10b981', fontWeight: 600 }}>Active</span>
                            </div>
                            <div className="arch-diagram" style={{ background: '#05080f', borderRadius: '12px', overflow: 'hidden' }}>
                                <object type="image/svg+xml" data="/branding/architecture_animated_v2.svg" style={{ width: '100%', height: 'auto', display: 'block' }}></object>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* About / Footer Section */}
            <footer id="about" style={{ padding: isMobile ? '3rem 1.5rem' : '4rem 2rem', background: '#020408', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr', gap: '4rem', alignItems: 'center' }}>
                    
                    {/* Creator Info */}
                    <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '2rem' }}>
                            <img src="/branding/qr.png" alt="QR" style={{ width: '80px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)' }} />
                            <div>
                                <h3 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.2rem' }}>Ambuj Kumar Tripathi</h3>
                                <p style={{ color: '#10b981', fontWeight: 500 }}>AI Engineer & RAG Specialist</p>
                            </div>
                        </div>
                        <p style={{ color: '#9ca3af', lineHeight: 1.6, marginBottom: '2rem' }}>
                            Engineered for high accuracy and compliance. This system utilizes cutting-edge orchestration, semantic vector search, and strict constitutional guardrails.
                        </p>
                        <div style={{ display: 'flex', gap: '1rem' }}>
                            <a href="https://ambuj-ai-portfolio.vercel.app/" target="_blank" rel="noreferrer" className="btn-secondary" style={{ padding: '0.6rem 1.2rem', borderRadius: '8px', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                                View Portfolio <FiArrowRight />
                            </a>
                            <a href="https://github.com/Ambuj123-lab" target="_blank" rel="noreferrer" style={{ padding: '0.6rem 1.2rem', borderRadius: '8px', textDecoration: 'none', color: '#fff', display: 'inline-flex', alignItems: 'center', gap: '8px', background: '#111827' }}>
                                <FaGithub /> GitHub
                            </a>
                        </div>
                    </div>

                    {/* HuggingFace Tweet */}
                    <div style={{ background: 'rgba(255,255,255,0.02)', padding: '1.5rem', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                            <span style={{ fontSize: '20px' }}>??</span>
                            <h4 style={{ fontWeight: 600, color: '#e5e7eb' }}>Recognized by Hugging Face</h4>
                        </div>
                        <blockquote className="twitter-tweet" data-theme="dark" style={{ margin: 0 }}>
                            <p lang="en" dir="ltr">Meet Ambuj-Tripathi-Indian-Legal-Llama-GGUF: a specialized AI model fine-tuned for Indian law. A game-changer for legal tech in India. <a href="https://t.co/SkLzeaDgpE">pic.twitter.com/SkLzeaDgpE</a></p>&mdash; Hugging Models (@HuggingModels) <a href="https://x.com/HuggingModels/status/2044027666324697451">April 14, 2026</a>
                        </blockquote>
                    </div>

                </div>
                
                <div style={{ textAlign: 'center', marginTop: '4rem', paddingTop: '2rem', borderTop: '1px solid rgba(255,255,255,0.05)', color: '#6b7280', fontSize: '0.9rem' }}>
                    &copy; {new Date().getFullYear()} Ambuj Kumar Tripathi. All rights reserved.
                </div>
            </footer>
        </div>
    );
}
