import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chunker import MultilingualChunker
from dotenv import load_dotenv

load_dotenv()

def ingest_data(file_path: str, index_path: str = "vectorstore"):
    print(f"Loading data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Chunking data...")
    # カスタムチャンカーの適用（ここで言語に合わせた適切な形態素分割が行われる）
    chunker = MultilingualChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split_text(text)

    from langchain.schema import Document
    docs = [Document(page_content=chunk) for chunk in chunks]

    print("Generating embeddings and saving to FAISS...")
    # 軽量でノートPCのリソースを圧迫しないよう、ローカルモデルではなくGeminiのEmbedding APIを使用
    # ベクトルDBには非常に軽量でメモリ効率が良いFAISSを採用
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    print(f"Ingestion complete. Index saved to {index_path}")

if __name__ == "__main__":
    # サンプルデータの作成と取り込み
    sample_text_path = "sample_data.txt"
    if not os.path.exists(sample_text_path):
        with open(sample_text_path, "w", encoding="utf-8") as f:
            f.write(
                "Artificial Intelligence is transforming the world and enabling new technologies in Canada. \n"
                "人工知能は世界を変え、社会に新たな価値を提供しています。日本の技術力と融合することでさらなる発展が見込まれます。\n"
                "人工智慧正在改變世界，並為台灣的半導體產業帶來前所未有的機遇與挑戰。"
            )
    
    # APIキーが設定されているかチェック
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY is not set in .env file.")
    else:
        ingest_data(sample_text_path)
