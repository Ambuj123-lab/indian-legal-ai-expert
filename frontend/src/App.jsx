import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Chat from './components/Chat';
import AdminPanel from './components/AdminPanel';
import Login from './pages/Login';
import AuthCallback from './pages/AuthCallback';
import { FiX, FiUser, FiLogOut } from 'react-icons/fi';

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

function Dashboard() {
  const { user, isAdmin, logout } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const isMobile = useIsMobile();

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
            {user?.picture ? <img src={user.picture} alt="User" /> : <FiUser size={18} />}
          </div>
          <div className="sidebar-user-info">
            <span className="sidebar-name">{user?.name || 'Legal Expert'}</span>
            <span className="sidebar-email">{user?.email}</span>
          </div>
        </div>

        <div className="sidebar-scrollable-content">
          <AdminPanel isAdmin={isAdmin} />
        </div>

        <button className="logout-btn" onClick={logout}>
          <FiLogOut size={16} /> <span>Sign Out</span>
        </button>
      </div>
    </div>
  );
}

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
