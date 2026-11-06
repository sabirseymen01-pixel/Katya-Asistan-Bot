FROM python:3.11-slim

# 1. Ses işleme, STT ve derleme için gerekli Linux sistem paketleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Çalışma dizini
WORKDIR /app

# 3. Bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# 4. Proje dosyalarını kopyala
COPY . .

# 5. Render dinamik port desteği ile başlat
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
