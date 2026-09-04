# マルチステージビルドを採用し、最終的なイメージサイズを極限まで小さくします
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

# ビルド依存関係をインストールし、ユーザースペースにパッケージを配置
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --user --no-cache-dir -r requirements.txt

# --- ランタイムステージ ---
FROM python:3.11-slim AS runtime

# メモリやCPUに制限がある古いノートPC向けの最適化設定
# スレッド数を制限し、コンテキストスイッチのオーバーヘッドを減らす
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

# ビルダーからインストール済みのパッケージのみをコピー (コンパイラ等は含まれないため軽量化される)
COPY --from=builder /root/.local /root/.local

# アプリケーションコードのコピー
COPY chunker.py ingest.py app.py ./

CMD ["python", "app.py"]
