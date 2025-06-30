FROM python:3.11-slim

WORKDIR /app

# NOMIC key = nk-fXeKDhexpRYcUUK44HzjbcPTzktZxp7uvcTqT_SlTEY

RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*



COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN nomic login nk-fXeKDhexpRYcUUK44HzjbcPTzktZxp7uvcTqT_SlTEY

COPY . .

EXPOSE 8000
RUN chmod +x ./cmd.sh
CMD ["./cmd.sh"]
