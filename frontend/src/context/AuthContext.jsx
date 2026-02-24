import { createContext, useState, useContext, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { jwtDecode } from 'jwt-decode';

const AuthContext = createContext(null);
const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [isAdmin, setIsAdmin] = useState(false);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();

    // Fetch admin status from backend
    async function fetchAdminStatus(token) {
        try {
            const res = await fetch(`${API_BASE}/api/me`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) {
                const data = await res.json();
                setIsAdmin(data.is_admin || false);
            }
        } catch {
            setIsAdmin(false);
        }
    }

    useEffect(() => {
        const token = localStorage.getItem('token');
        if (token) {
            try {
                const decoded = jwtDecode(token);
                // Check expiry
                if (decoded.exp * 1000 > Date.now()) {
                    setUser({
                        email: decoded.email,
                        name: decoded.name,
                        picture: decoded.picture,
                    });
                    fetchAdminStatus(token);
                } else {
                    localStorage.removeItem('token');
                }
            } catch {
                localStorage.removeItem('token');
            }
        }
        setLoading(false);
    }, []);

    const login = useCallback((token) => {
        localStorage.setItem('token', token);
        const decoded = jwtDecode(token);
        setUser({
            email: decoded.email,
            name: decoded.name,
            picture: decoded.picture,
        });
        setLoading(false);
        fetchAdminStatus(token);
    }, []);

    const logout = useCallback(async () => {
        // Call backend to cleanup temp vectors from Qdrant before clearing local state
        try {
            const token = localStorage.getItem('token');
            if (token) {
                await fetch(`${API_BASE}/auth/logout`, {
                    method: 'POST',
                    headers: { Authorization: `Bearer ${token}` },
                });
            }
        } catch {
            // Non-critical — proceed with local logout even if backend call fails
        }
        localStorage.removeItem('token');
        setUser(null);
        setIsAdmin(false);
        navigate('/login');
    }, [navigate]);

    return (
        <AuthContext.Provider value={{ user, isAdmin, login, logout, loading }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    return useContext(AuthContext);
}
