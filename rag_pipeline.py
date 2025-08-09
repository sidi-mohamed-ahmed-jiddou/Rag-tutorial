import os
from dotenv import load_dotenv
import warnings

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

import chromadb

# Ignore certains avertissements inutiles
warnings.filterwarnings("ignore", category=FutureWarning)

# Charger les variables d’environnement
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Sécurité : s'assurer que la clé API est bien chargée
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY est manquant dans .env")

# 1. Configurer le LLM (Groq) et les embeddings (HuggingFace)
Settings.llm = Groq(
    api_key=GROQ_API_KEY,
    model="llama3-70b-8192",
    temperature=0.2,
    max_tokens=2048,
)

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Charger les documents
documents = SimpleDirectoryReader("data").load_data()

Settings.chunk_size = 512
Settings.chunk_overlap = 50

if not documents:
    raise ValueError("Aucun document trouvé dans le dossier 'data'")

# Initialiser ChromaDb
db = chromadb.PersistentClient(path="database")
chroma_collection = db.get_or_create_collection("database-collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if os.path.exists("database"):
    print("Loading index from storage...")
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    print("Index loaded")

else:
    print("Creating index...")
    # Construire l’index AVEC le storage_context (important !)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    print("Index creation complete")

# 5. Configurer le retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10
)

# 6. Assembler le query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
)

# 7. Boucle de requête interactive
while True:
    query = input("Question: ")
    if query.lower() in {"exit", "quit"}:
        break

    response = query_engine.query(query)
    print(f"\n Answer : {response}\n")