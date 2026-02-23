import { useEffect, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function AuthCallback() {
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const { login } = useAuth();
    const hasRun = useRef(false);

    useEffect(() => {
        if (hasRun.current) return;
        hasRun.current = true;

        const token = searchParams.get('token');
        const error = searchParams.get('error');

        if (token) {
            login(token);
            navigate('/', { replace: true });
        } else if (error) {
            navigate('/login?error=' + error, { replace: true });
        } else {
            navigate('/login', { replace: true });
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
