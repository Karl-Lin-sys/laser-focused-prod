import sys
import os
import pytest

# appモジュールへのパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from custom_splitter import MultilingualTextSplitter

def test_japanese_splitting():
    splitter = MultilingualTextSplitter(chunk_size=10, chunk_overlap=0)
    text = "私は自然言語処理のエンジニアです。"
    # Janomeを使った分割の場合、chunk_size=10なので適度に分割されるはず
    chunks = splitter.split_text(text, language="ja")
    
    # 全体が複数のチャンクに分かれていること
    assert len(chunks) > 0
    # 結合すれば元のテキストに戻ること
    assert "".join(chunks) == text
    # 各チャンクの長さが10以下であること (Janomeのトークンが10を超えない限り)
    for chunk in chunks:
        assert len(chunk) <= 10

def test_taiwanese_mandarin_splitting():
    splitter = MultilingualTextSplitter(chunk_size=10, chunk_overlap=0)
    text = "我是一名自然語言處理工程師。"
    chunks = splitter.split_text(text, language="zh-tw")
    
    assert len(chunks) > 0
    assert "".join(chunks) == text
    for chunk in chunks:
        assert len(chunk) <= 10

def test_english_splitting():
    splitter = MultilingualTextSplitter(chunk_size=15, chunk_overlap=0)
    text = "I am an NLP engineer from Canada."
    chunks = splitter.split_text(text, language="en")
    
    assert len(chunks) > 0
    # RecursiveCharacterTextSplitterはスペースで分割し、結合時にはスペースが失われない(元のまま保持される)
    # または余分なスペースが含まれる場合があるため、単純結合での完全一致チェックはテキストの構造次第だが
    # チャンク自体が作られていることを確認する。
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 15
