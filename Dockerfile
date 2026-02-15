# ==================================
# Stage 1: Build Frontend (React)
# ==================================
FROM node:18-alpine as frontend-build

WORKDIR /app/frontend

# Install dependencies
COPY frontend/package.json ./
COPY frontend/package-lock.json ./
RUN npm ci

# Build the project
COPY frontend/ ./
ENV VITE_API_URL=""
RUN npm run build


# ==================================
# Stage 2: Setup Backend (FastAPI)
# ==================================
FROM python:3.11-slim as backend

WORKDIR /app

# Install system dependencies (needed for compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies - Use --no-cache-dir to keep image small
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend

# Copy built frontend assets from Stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose port (Koyeb uses 8000 by default for web services)
EXPOSE 8000

# Set working directory to backend so "app" package is found
WORKDIR /app/backend

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
