/**
 * API Client — All backend communication
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function getToken() {
    return localStorage.getItem('token');
}

async function authFetch(url, options = {}) {
    const token = getToken();
    const headers = { ...options.headers };
    if (token) headers['Authorization'] = `Bearer ${token}`;
    if (!(options.body instanceof FormData)) {
        headers['Content-Type'] = 'application/json';
    }

    const res = await fetch(`${API_BASE}${url}`, { ...options, headers });
    if (res.status === 401) {
        localStorage.removeItem('token');
        window.location.href = '/login';
        throw new Error('Unauthorized');
    }
    return res;
}

// --- Auth ---
export function getLoginUrl() {
    return `${API_BASE}/auth/login`;
}

export async function logout() {
    try { await authFetch('/auth/logout', { method: 'POST' }); } catch { }
    localStorage.removeItem('token');
}

// --- Chat ---
export async function sendChat(question) {
    const res = await authFetch('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ question }),
    });
    return res.json();
}

export async function streamChat(question, onToken, onDone) {
    const token = getToken();
    const res = await fetch(`${API_BASE}/api/chat/stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ question, use_streaming: true }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                try {
                    const data = JSON.parse(line.slice(6));
                    if (data.done) {
                        onDone(data);
                    } else if (data.token) {
                        onToken(data.token);
                    }
                } catch { }
            }
        }
    }
}

// --- History ---
export async function getHistory() {
    const res = await authFetch('/api/history');
    return res.json();
}

export async function clearHistory() {
    const res = await authFetch('/api/history', { method: 'DELETE' });
    return res.json();
}

// --- Feedback ---
export async function submitFeedback(question, response, rating) {
    const res = await authFetch('/api/feedback', {
        method: 'POST',
        body: JSON.stringify({ question, response, rating }),
    });
    return res.json();
}

// --- Admin ---
export async function syncDocuments() {
    const res = await authFetch('/api/admin/documents/sync', { method: 'POST' });
    return res.json();
}

export async function listDocuments() {
    const res = await authFetch('/api/admin/documents');
    return res.json();
}

export async function deleteDocument(fileName) {
    const res = await authFetch(`/api/admin/documents/${encodeURIComponent(fileName)}`, { method: 'DELETE' });
    return res.json();
}

// --- Stats ---
export async function getStats() {
    const res = await authFetch('/api/stats');
    return res.json();
}

// --- User Temp Uploads ---
export async function uploadTempFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await authFetch('/api/upload', {
        method: 'POST',
        body: formData,
    });
    return res.json();
}

export async function listTempUploads() {
    const res = await authFetch('/api/uploads');
    return res.json();
}

export async function deleteTempUpload(fileName) {
    const res = await authFetch(`/api/uploads/${encodeURIComponent(fileName)}`, { method: 'DELETE' });
    return res.json();
}

// --- Health ---
export async function healthCheck() {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
}
