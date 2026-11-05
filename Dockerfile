FROM python:3.11-slim

# 1. Ses ve derleme için gerekli Linux sistem paketlerini kur
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 2. Çalışma dizini
WORKDIR /app

# 3. Paketleri yükle
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Proje dosyalarını kopyala
COPY . .

# 5. Render port değişkenini tanımla ve uygulamayı başlat
ENV PORT=10000
EXPOSE 10000

CMD ["python3", "main.py"]
