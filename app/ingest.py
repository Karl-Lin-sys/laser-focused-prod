import os
import glob
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from custom_splitter import MultilingualTextSplitter

def load_documents(data_dir: str):
    """
    指定されたディレクトリからテキストファイルを読み込む
    """
    documents = []
    for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            # ファイル名からメタデータを簡易的に作成
            documents.append(Document(page_content=text, metadata={"source": filepath}))
    return documents

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    db_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    
    # ドキュメントの読み込み
    docs = load_documents(data_dir)
    if not docs:
        print("No documents found in data directory.")
        return

    # ここでは例として日本語を想定 (メタデータ等から動的に言語を判定することも可能)
    language = "ja"  
    
    # カスタムチャンカーを用いて分割
    splitter = MultilingualTextSplitter(chunk_size=500, chunk_overlap=0)
    split_docs = splitter.split_documents(docs, language=language)
    
    print(f"Loaded {len(docs)} documents and split into {len(split_docs)} chunks.")

    # 計算リソースを抑えるため、ローカルではなくOpenAIのAPIを利用
    embeddings = OpenAIEmbeddings()
    
    # ChromaDBに保存 (軽量なベクトルデータベース)
    print("Ingesting into ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings, 
        persist_directory=db_dir
    )
    vectorstore.persist()
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
