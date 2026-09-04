from chunker import MultilingualChunker

def test_japanese_chunking():
    chunker = MultilingualChunker(chunk_size=20, chunk_overlap=0)
    text = "私は自然言語処理と機械学習が大好きです。"
    chunks = chunker.split_text(text, lang="ja")
    
    assert len(chunks) > 0
    # チャンク内に形態素分割用の一時的なスペースが残っていないかテスト
    assert " " not in chunks[0]
    
def test_chinese_chunking():
    chunker = MultilingualChunker(chunk_size=20, chunk_overlap=0)
    text = "人工智慧正在改變世界。"
    chunks = chunker.split_text(text, lang="zh-tw")
    
    assert len(chunks) > 0
    assert " " not in chunks[0]

def test_english_chunking():
    chunker = MultilingualChunker(chunk_size=30, chunk_overlap=0)
    text = "I love natural language processing."
    chunks = chunker.split_text(text, lang="en")
    
    assert len(chunks) > 0
