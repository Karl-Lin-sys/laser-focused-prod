# 軽量化とビルド速度向上のためのマルチステージビルド
FROM python:3.10-slim AS builder

WORKDIR /build

COPY requirements.txt .
# 古いノートPCでもビルドが速くなるよう、キャッシュを利用して依存関係をインストール
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

# builderステージからインストール済みのパッケージをコピー
COPY --from=builder /root/.local /root/.local
COPY . /app

# パスを通す
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# アプリケーションの実行
CMD ["python", "app/app.py"]
