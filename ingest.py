import os
import jieba
from janome.tokenizer import Tokenizer
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import TextSplitter

class AsianLanguageTextSplitter(TextSplitter):
    """
    カスタムテキスト分割クラス。
    英語のスペース区切りとは異なり、日本語や台湾華語の形態素を破壊しないよう、
    専用のトークナイザ（Janome, Jieba）を用いて単語単位で分割し、文字数制約内でチャンク化します。
    これにより検索精度（RAGの検索フェーズ）が向上します。
    """
    def __init__(self, language="ja", chunk_size=500, chunk_overlap=50, **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.language = language
        if language == "ja":
            self.janome_tokenizer = Tokenizer()

    def split_text(self, text: str) -> list[str]:
        # 言語ごとの形態素解析を用いたトークン化
        if self.language == "zh-tw":
            tokens = list(jieba.cut(text))
        elif self.language == "ja":
            tokens = [token.surface for token in self.janome_tokenizer.tokenize(text)]
        else: # 英語など（スペース区切り）
            tokens = text.split(" ")

        chunks = []
        current_chunk = []
        current_length = 0

        # トークン（形態素）単位で文字数をカウントしチャンク化
        for token in tokens:
            token_len = len(token)
            if current_length + token_len > self._chunk_size and current_chunk:
                chunks.append("".join(current_chunk) if self.language != "en" else " ".join(current_chunk))
                # オーバーラップの実装（簡易的に最後の数トークンを残すアプローチも可能ですが、
                # 本実装では軽量化のためリセット方式を採用しています）
                current_chunk = []
                current_length = 0
            
            current_chunk.append(token)
            current_length += token_len

        if current_chunk:
            chunks.append("".join(current_chunk) if self.language != "en" else " ".join(current_chunk))

        return chunks

def ingest_data():
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"{data_dir} directory created. Please put your .txt files here.")
        return

    # データの読み込み
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print("No documents found in ./data")
        return

    # ここでは例として日本語(ja)を指定。必要に応じて zh-tw や en に切り替え可能。
    splitter = AsianLanguageTextSplitter(language="ja", chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    # Embeddingモデルの初期化（ローカルのメモリを圧迫しないようGemini APIを利用）
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 軽量なChromaDBに保存して永続化
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Ingestion completed successfully.")

if __name__ == "__main__":
    ingest_data()
