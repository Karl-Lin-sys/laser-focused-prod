from langchain.text_splitter import RecursiveCharacterTextSplitter
import jieba
from janome.tokenizer import Tokenizer

class MultilingualTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.janome_tokenizer = Tokenizer()
        
    def split_text(self, text: str, language: str = "en") -> list[str]:
        """
        Design Intent (設計意図):
        英語（カナダ英語など）ではスペース区切りで自然に単語の境界を認識できますが、
        日本語や台湾華語（繁体字）ではスペースがないため、標準のチャンカー（文字数ベース）では
        形態素（意味の最小単位）の途中で強制的に分割されてしまうリスクがあります。
        形態素が破壊されると、ベクトル化の際の文脈の喪失や、検索精度の低下を招きます。
        
        そのため、各言語に特化した形態素解析器を用いてまず単語レベルに分割し、
        それらを意味を保ったまま指定のchunk_sizeに収まるように再結合（マージ）するカスタムロジックを採用しています。
        - 日本語: Janome (軽量でpure Pythonのため環境構築が容易)
        - 台湾華語: Jieba (繁体字にも対応可能、広く使われている)
        """
        if language == "ja":
            # Janomeを使用して形態素に分割
            tokens = [token.surface for token in self.janome_tokenizer.tokenize(text)]
        elif language == "zh-tw":
            # Jiebaを使用して形態素に分割
            tokens = list(jieba.cut(text, cut_all=False))
        else:
            # 英語などの場合は標準のRecursiveCharacterTextSplitterを使用
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            return splitter.split_text(text)

        # 形態素をチャンクにまとめる処理
        chunks = []
        current_chunk = ""
        
        for token in tokens:
            if len(current_chunk) + len(token) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = token
            else:
                current_chunk += token
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def split_documents(self, documents, language: str = "en"):
        # LangChainのDocumentオブジェクトのリストを受け取り、分割する
        from langchain.schema import Document
        split_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content, language=language)
            for chunk in chunks:
                split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
        return split_docs
