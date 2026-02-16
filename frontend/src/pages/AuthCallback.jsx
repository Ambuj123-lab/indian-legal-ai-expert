import { useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function AuthCallback() {
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const { login } = useAuth();

    useEffect(() => {
        const token = searchParams.get('token');
        const error = searchParams.get('error');

        if (token) {
            login(token);
            window.location.href = '/';
        } else if (error) {
            navigate('/login?error=' + error);
        } else {
            navigate('/login');
        }
    }, [searchParams, login, navigate]);

    return (
        <div className="login-page">
            <div className="login-card">
                <div className="spinner large" />
                <p style={{ marginTop: '1rem', color: '#8b8fa3' }}>Authenticating...</p>
            </div>
        </div>
    );
}
