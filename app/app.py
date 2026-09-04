import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

def build_qa_chain():
    db_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    
    # 計算リソースを抑えるためOpenAIのEmbeddingとLLMを使用
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    
    # 軽量な推論を行うためにGPT-3.5-turbo (またはgpt-4o-mini) などの軽量モデルを指定
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # 検索チェーンの構築
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain

def main():
    print("Initializing RAG system...")
    qa_chain = build_qa_chain()
    print("System ready. Type 'exit' to quit.")
    
    while True:
        query = input("\nQuery: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        if not query.strip():
            continue
            
        try:
            result = qa_chain.invoke({"query": query})
            print("\n--- Answer ---")
            print(result["result"])
            print("--------------")
            
            print("\nSources:")
            for doc in result["source_documents"]:
                print(f"- {doc.metadata.get('source', 'Unknown')}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
