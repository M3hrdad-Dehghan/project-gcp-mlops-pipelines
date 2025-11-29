# ===============================
# 1) Base Image (Python 3.10 Slim)
# ===============================
FROM python:3.10-slim

# -------------------------------
# 2) Install system dependencies
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# 3) Create working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# 4) Copy dependency files
# -------------------------------
COPY requirements.txt .
COPY ./static /app/static

# -------------------------------
# 5) Install Python dependencies
# -------------------------------
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -------------------------------
# 6) Copy application code
# -------------------------------
COPY . .

# -------------------------------
# 7) Expose API port
# -------------------------------
# Use 8000 so the `docker run -p 8000:8000` command from README works
EXPOSE 8000

# -------------------------------
# 8) Start FastAPI with uvicorn
# -------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
