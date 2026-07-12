FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the GGUF model (~0.39 GB)
RUN mkdir -p /app/models && \
    curl -L -o /app/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
    "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

# Copy application code
COPY . .

# Expose Flask port (for web mode)
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Fix line endings and make entrypoint executable
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

# Run the application through the entrypoint
ENTRYPOINT ["./entrypoint.sh"]
