import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Chat from './components/Chat';
import AdminPanel from './components/AdminPanel';
import Login from './pages/Login';
import AuthCallback from './pages/AuthCallback';

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
        <Chat />
      </div>

      {/* Mobile sidebar toggle */}
      {isMobile && (
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          style={{
            position: 'fixed',
            bottom: '80px',
            right: '16px',
            zIndex: 100,
            width: '44px',
            height: '44px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #6c5ce7, #a29bfe)',
            border: 'none',
            color: 'white',
            fontSize: '1.2rem',
            cursor: 'pointer',
            boxShadow: '0 4px 15px rgba(108, 92, 231, 0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
          title="Toggle Admin Panel"
        >
          {sidebarOpen ? '✕' : '⚙'}
        </button>
      )}

      {/* Sidebar: always visible on desktop, toggled on mobile */}
      {(!isMobile || sidebarOpen) && (
        <div className="dashboard-sidebar" style={isMobile ? {
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          maxHeight: '60vh',
          zIndex: 99,
          borderTop: '1px solid rgba(255,255,255,0.1)',
          boxShadow: '0 -4px 20px rgba(0,0,0,0.5)'
        } : {}}>
          <AdminPanel isAdmin={isAdmin} />
          <button className="logout-btn" onClick={logout}>Sign Out</button>
        </div>
      )}
    </div>
  );
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/auth/callback" element={<AuthCallback />} />
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
