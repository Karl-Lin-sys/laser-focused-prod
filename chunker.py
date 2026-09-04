import jieba
from janome.tokenizer import Tokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MultilingualChunker:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 日本語用形態素解析器 (軽量で環境依存が少ないJanomeを採用)
        self.ja_tokenizer = Tokenizer()
        
        # 英語・デフォルト用の標準チャンカー
        self.en_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _is_japanese(self, text: str) -> bool:
        # 簡易的なヒューリスティック: ひらがな・カタカナが含まれているか
        for char in text:
            if '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF':
                return True
        return False

    def _is_chinese(self, text: str) -> bool:
        # 漢字の範囲をチェック
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def split_text(self, text: str, lang: str = "auto") -> list[str]:
        """
        【設計意図】
        英語(カナダ英語など)のように単語がスペースで区切られている言語は、
        標準の RecursiveCharacterTextSplitter で意味を損なわずチャンク化できます。
        しかし、日本語や台湾華語のようにスペース区切りを持たない言語では、
        指定文字数で強制的に分割すると単語(形態素)の中間や意味の区切りが破壊され、
        ベクトル検索時の精度が著しく低下します。
        
        このカスタム戦略では、Janome(日本語)やJieba(台湾華語)を用いて形態素/単語レベルで
        トークナイズし、一時的に単語間にスペースを挿入します。
        スペース区切り状態にした後で RecursiveCharacterTextSplitter に渡すことで、
        Splitterはスペースを区切り文字として認識し、形態素を破壊せずに指定サイズに収めます。
        チャンク化完了後、挿入したスペースを除去し、元の自然な言語形式に戻します。
        """
        if lang == "auto":
            if self._is_japanese(text):
                lang = "ja"
            elif self._is_chinese(text):
                lang = "zh-tw"
            else:
                lang = "en"

        if lang == "ja":
            # Janomeによる形態素解析
            tokens = [token.surface for token in self.ja_tokenizer.tokenize(text)]
            # RecursiveCharacterTextSplitterが処理できるようスペースで結合
            spaced_text = " ".join(tokens)
            chunks = self.en_splitter.split_text(spaced_text)
            # チャンク化後にスペースを除去
            return [chunk.replace(" ", "") for chunk in chunks]

        elif lang == "zh-tw":
            # Jiebaによる単語分割 (台湾華語向け)
            tokens = list(jieba.cut(text, cut_all=False))
            spaced_text = " ".join(tokens)
            chunks = self.en_splitter.split_text(spaced_text)
            return [chunk.replace(" ", "") for chunk in chunks]

        else:
            # カナダ英語などの場合はそのまま処理
            return self.en_splitter.split_text(text)
