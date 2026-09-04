import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def generate_answer(query: str):
    # APIキーが設定されているか確認
    if not os.environ.get("GEMINI_API_KEY"):
        return "Error: GEMINI_API_KEY environment variable not set."

    # EmbeddingとローカルのChromaDBをロード
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ローカルリソースを節約するため、LLMとしてGemini APIを使用
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # 多言語に対応するためのプロンプトテンプレート
    system_prompt = (
        "You are an AI assistant skilled in English, Japanese, and Taiwanese Mandarin. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Answer in the same language as the question.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # RAGチェーンの構築
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    return response["answer"]

if __name__ == "__main__":
    print("--- Multilingual RAG System ---")
    query = input("Enter your question: ")
    answer = generate_answer(query)
    print("\n[Answer]:")
    print(answer)
