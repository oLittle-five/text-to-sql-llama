FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Expose port
EXPOSE 8000

# Environment variables (override at runtime)
ENV BASE_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
ENV ADAPTER_ID=oLittle-five/llama3-8b-wikisql-qlora
ENV QUANTIZE_4BIT=true
ENV MAX_NEW_TOKENS=128
ENV REPETITION_PENALTY=1.3

# Run the API
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
