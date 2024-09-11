#!/usr/bin/env python
# coding: utf-8

# pip install scikit-learn pandas pyarrow
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SKLearnVectorStore

load_dotenv()

loader = TextLoader('data_sources/HistoriaEspana.txt', encoding="utf8")
documents = loader.load()

# Dividir en chunks basándose en tokens
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)

# Conectar a OpenAI para los embeddings
embedding_function = OpenAIEmbeddings()

# Alternativa con SKLearn Vector Store
persist_path = "./ejemplos_embedding_db"

# Creamos la DB de vectores a partir de los documentos y la función embeddings
vector_store = SKLearnVectorStore.from_documents(
    documents=docs,
    embedding=embedding_function,
    persist_path=persist_path,
    serializer="parquet",  # el serializador o formato de la BD lo definimos como parquet
)

vector_store.persist()
# Creamos un nuevo documento que será nuestra "consulta" para buscar el de mayor similitud en nuestra Base de Datos
# de Vectores y devolverlo
consulta = "dame información de la Primera Guerra Mundial"
docs = vector_store.similarity_search(consulta)
print(docs[0].page_content)

# ## Cargar la BD de vectores (uso posterior una vez tenemos creada ya la BD)
vector_store_connection = SKLearnVectorStore(
    embedding=embedding_function, persist_path=persist_path, serializer="parquet"
)
print("Una instancia de la DB de vectores se ha cargado desde ", persist_path)

nueva_consulta = "¿Qué paso en el siglo de Oro?"
docs = vector_store_connection.similarity_search(nueva_consulta)
print(docs[0].page_content)
