import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Chat from './components/Chat';
import AdminPanel from './components/AdminPanel';
import Login from './pages/Login';
import AuthCallback from './pages/AuthCallback';
import { FiX, FiUser, FiLogOut, FiCode } from 'react-icons/fi';

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth <= 768);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  return isMobile;
}

function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="login-page"><div className="spinner large" /></div>;
  if (!user) return <Navigate to="/login" />;
  return children;
}

  const Dashboard = () => {
    const { user, isAdmin, logout } = useAuth();
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [imgError, setImgError] = useState(false);
    const isMobile = useIsMobile();

    // Reset error when user changes
    useEffect(() => {
      setImgError(false);
    }, [user]);

    return (
      <div className="dashboard">
        <div className="dashboard-main">
          <Chat sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} isMobile={isMobile} />
        </div>

        {/* Overlay to close sidebar on mobile */}
        {isMobile && sidebarOpen && (
          <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)}></div>
        )}

        {/* Sidebar: always visible on desktop, toggled via CSS on mobile */}
        <div className={`dashboard-sidebar ${isMobile ? (sidebarOpen ? 'open' : 'closed') : ''}`}>
          {isMobile && (
            <div className="sidebar-mobile-header">
              <span className="sidebar-mobile-title">Menu</span>
              <button className="sidebar-close-btn" onClick={() => setSidebarOpen(false)}>
                <FiX size={20} />
              </button>
            </div>
          )}

          <div className="sidebar-user-summary">
            <div className="sidebar-user-avatar">
              {user?.picture && !imgError ? (
                <img 
                  src={user.picture} 
                  alt="User" 
                  referrerPolicy="no-referrer"
                  onError={() => setImgError(true)} 
                />
              ) : (
                <div className="sidebar-user-initial">
                  {user?.name?.[0]?.toUpperCase() || <FiUser size={18} />}
                </div>
              )}
            </div>
            <div className="sidebar-user-info">
              <div className="sidebar-name-row">
                <span className="sidebar-name">{user?.name || 'Legal Expert'}</span>
              </div>
              <div className="sidebar-email-row">
                <span className="sidebar-email">{user?.email}</span>
              </div>
            </div>
          </div>

          <div className="sidebar-scrollable-content">
            <AdminPanel isAdmin={isAdmin} />
          </div>

          {/* Professional Footer Branding — Matching Navbar Aesthetic */}
          <div className="sidebar-footer-branding">
            Built by <span className="footer-name">Ambuj Kumar Tripathi</span> 
            <span className="footer-sep"> · </span> 
            <span className="footer-role">RAG Systems Architect</span> 
            <span className="footer-sep"> · </span> 
            <a href="https://ambuj-portfolio-v2.netlify.app" target="_blank" rel="noopener noreferrer" className="footer-link">ambuj-portfolio-v2.netlify.app</a>
          </div>

          <button className="logout-btn" onClick={logout}>
            <FiLogOut size={16} /> <span>Sign Out</span>
          </button>
        </div>
      </div>
    );
  };

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/auth-callback" element={<AuthCallback />} />
      <Route path="/" element={
        <ProtectedRoute>
          <Dashboard />
        </ProtectedRoute>
      } />
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}
