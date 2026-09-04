import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def setup_rag():
    index_path = "vectorstore"
    if not os.path.exists(index_path):
        raise FileNotFoundError("FAISS index not found. Please run ingest.py first.")

    # FAISSを用いた軽量なローカル検索
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # allow_dangerous_deserialization=True はローカルで作成した信頼できるpickleをロードするために必要
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 計算リソースを抑えるため、回答生成にはGemini APIを利用
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0)

    system_prompt = (
        "You are an intelligent assistant. Use the following pieces of retrieved context to answer the question.\n"
        "Please respond in the same language as the user's question (e.g., Canadian English, Japanese, or Taiwanese Mandarin).\n"
        "If you don't know the answer, just say that you don't know.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY is not set in .env file.")
        return

    print("Initializing Lightweight Multi-lingual RAG system...")
    try:
        rag_chain = setup_rag()
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    print("System ready. Type 'exit' or 'quit' to end the session.")
    while True:
        try:
            query = input("\nUser: ")
            if query.lower() in ['exit', 'quit']:
                break
            
            response = rag_chain.invoke({"input": query})
            print(f"\nAI: {response['answer']}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError processing query: {e}")

if __name__ == "__main__":
    main()
