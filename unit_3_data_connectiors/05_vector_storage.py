#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías

# In[ ]:


from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader


# ### Carga de documento y split

# In[ ]:


# Cargar el documento
loader = TextLoader('data_sources/HistoriaEspana.txt', encoding="utf8")
documents = loader.load()


# In[ ]:


# Dividir en chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500) #Otro método de split basándose en tokens
docs = text_splitter.split_documents(documents)


# ### Conectar a OpenAI para los embeddings

# In[ ]:


f = open('../OpenAI_key.txt')
api_key = f.read()


# In[ ]:


funcion_embedding = OpenAIEmbeddings(openai_api_key=api_key)


# # Alternativa con SKLearn Vector Store

# In[ ]:


from langchain_community.vectorstores import SKLearnVectorStore #pip install scikit-learn / pip install pandas pyarrow


# In[ ]:


persist_path="./ejemplosk_embedding_db"  #ruta donde se guardará la BBDD vectorizada

#Creamos la BBDD de vectores a partir de los documentos y la función embeddings
vector_store = SKLearnVectorStore.from_documents(
    documents=docs,
    embedding=funcion_embedding,
    persist_path=persist_path,
    serializer="parquet", #el serializador o formato de la BD lo definimos como parquet
)


# In[ ]:


# Fuerza a guardar los nuevos embeddings en el disco
vector_store.persist()


# In[ ]:


#Creamos un nuevo documento que será nuestra "consulta" para buscar el de mayor similitud en nuestra Base de Datos de Vectores y devolverlo
consulta = "dame información de la Primera Guerra Mundial"
docs = vector_store.similarity_search(consulta)
print(docs[0].page_content)


# ## Cargar la BD de vectores (uso posterior una vez tenemos creada ya la BD)

# In[ ]:


vector_store_connection = SKLearnVectorStore(
    embedding=funcion_embedding, persist_path=persist_path, serializer="parquet"
)
print("Una instancia de la BBDD de vectores se ha cargado desde ", persist_path)


# In[ ]:


vector_store_connection


# In[ ]:


nueva_consulta = "¿Qué paso en el siglo de Oro?"


# In[ ]:


docs = vector_store_connection.similarity_search(nueva_consulta)
print(docs[0].page_content)


# In[ ]:





# # Alternativa con ChromaDB

# In[ ]:


import chromadb #pip install chromadb en una terminal
from langchain_chroma import Chroma #pip install langchain_chroma en una terminal


# In[ ]:


# Cargar en ChromaDB
#db = Chroma.from_documents(docs, funcion_embedding,collection_name="langchain",persist_directory='./ejemplo_embedding_db')
#Se crean en el directorio persistente la carpeta con los vectores y otra con las string, aparte de una carpeta "index" que mapea vectores y strings


# In[ ]:


# Fuerzar a guardar los nuevos embeddings en el disco
db.persist()


# ### Cargar los Embeddings desde el disco creando la conexión a ChromaDB

# In[ ]:


db_connection = Chroma(persist_directory='./ejemplo_embedding_db/',embedding_function=funcion_embedding)


# In[ ]:


#Creamos un nuevo documento para buscar el de mayor similitud en nuestra Base de Datos de Vectores y devolverlo
nuevo_documento = "What did FDR say about the cost of food law?"


# In[ ]:


docs = db_connection.similarity_search(nuevo_documento)


# In[ ]:


#print(docs[0].page_content) #El primer elemento es el de mayor similitud, por defecto se devuelven hasta 4 vectores (k=4)


# ## Añadir nueva información a la BD de vectores

# In[ ]:


# Cargar documento y dividirlo
loader = TextLoader('data_sources/NuevoDocumento.txt', encoding="utf8")
documents = loader.load()


# In[ ]:


text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)


# In[ ]:


# Cargar en Chroma
db = Chroma.from_documents(docs, embedding_function, persist_directory='./ejemplo_embedding_db')


# In[ ]:


docs = db.similarity_search('insertar_nueva_búsqueda')


# In[ ]:


#docs[0].page_content


# In[ ]:





# In[ ]:




