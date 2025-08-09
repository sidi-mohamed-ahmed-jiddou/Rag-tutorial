from dotenv import load_dotenv
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Charger les variables d'environnement
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = Groq(
    api_key=GROQ_API_KEY,
    model="llama3-70b-8192",
    temperature=0.2,
    max_tokens=2048,
)

# Définir le modèle d'embedding local
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Charger les documents depuis un dossier
documents = SimpleDirectoryReader("data").load_data()

Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Créer l’index vectoriel
index = VectorStoreIndex.from_documents(documents)

# Créer le moteur de requête
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=5,
    verbose=True
)


# ChatEngine that maintains conversational memory.
# chat_query = index.as_chat_engine(llm=llm, verbose=True)

# Exécuter une requête
while True:
    user_query = input("Question: ")
    if user_query == "exit":
        break
    # chat_response = chat_query.chat(user_query)

    response = query_engine.query(user_query)
    print(f"Answer :  {response}")
    # print(f"Answer :  {chat_response}")



