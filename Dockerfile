# Official Python lightweight image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install essential OS dependencies for OpenCV + InsightFace + ONNXRuntime
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (use caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Streamlit config for cloud
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Vercel gives a dynamic port
ENV PORT=8501

# Expose port (for local use)
EXPOSE 8501

# RUN STREAMLIT
CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
