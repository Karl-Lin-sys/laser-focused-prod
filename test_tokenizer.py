from ingest import AsianLanguageTextSplitter

def test_japanese_tokenizer():
    # 日本語（Janome）でのチャンク化テスト
    splitter = AsianLanguageTextSplitter(language="ja", chunk_size=10, chunk_overlap=0)
    text = "私はAIエンジニアです。"
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 0
    # 「私」という形態素がそのまま残っているか確認
    assert "私" in chunks[0]

def test_chinese_tokenizer():
    # 台湾華語（Jieba）でのチャンク化テスト
    splitter = AsianLanguageTextSplitter(language="zh-tw", chunk_size=10, chunk_overlap=0)
    text = "我是一名人工智慧工程師。"
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 0
    # 「人工智慧」という単語が破壊されていないか確認
    combined = "".join(chunks)
    assert "人工智慧" in combined

def test_english_fallback():
    # 英語（フォールバック）でのチャンク化テスト
    splitter = AsianLanguageTextSplitter(language="en", chunk_size=10, chunk_overlap=0)
    text = "I am an AI engineer."
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 0
    assert "AI" in chunks[0] or "AI" in chunks[1]
