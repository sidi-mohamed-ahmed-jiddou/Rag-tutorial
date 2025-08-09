from llama_cloud_services import LlamaParse
from dotenv import load_dotenv
import os

# Chargement des variables d’environnement
load_dotenv()
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Initialiser le vector store et l’embedder une seule fois
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding,
    persist_directory="chroma_db",
)

# Prompt structuré pour l’assistant
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a helpful assistant that answers questions based on the provided context: {context}.
        If the answer is not in the context, say "I don't know." Do not make up answers.
    """),
    ("human", "{question}"),
])

# Initialiser le modèle Groq
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
)

def extract_text_from_pdf_with_llama(pdf_file_path: str) -> str:
    """
    Extrait le texte d'un fichier PDF en utilisant LlamaParse.
    """
    parser = LlamaParse(
        api_key=LLAMA_API_KEY,
        num_workers=4,
        verbose=True,
    )
    result = parser.parse(pdf_file_path)

    # Concatène le texte de toutes les pages
    return ''.join(page.text for page in result.pages)

def process_pdf_and_index(pdf_file_path: str):
    """
    Extrait le texte, le découpe, puis l'indexe dans ChromaDB.
    """
    text = extract_text_from_pdf_with_llama(pdf_file_path)
    documents = [Document(page_content=text)]

    # Découper le texte
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_documents = text_splitter.split_documents(documents)

    if not vector_store.get()['ids']:
        vector_store.add_documents(split_documents)

def main(user_query: str) -> str:
    """
    Traite une question utilisateur à l'aide des documents indexés.
    """
    relevant_docs = vector_store.similarity_search(user_query, k=4)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    response = (prompt | llm).invoke({
        "question": user_query,
        "context": context,
    })

    return response.content

if __name__ == "__main__":
    # Exemple d'utilisation
    pdf_path = "data/22365_3_Prompt Engineering_v7 (1).pdf"
    process_pdf_and_index(pdf_path)

    user_question = input("User question:  ")
    answer = main(user_question)
    print(f"Answer: {answer}")
