#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## 0. Importar librerías

# In[ ]:


from langchain.document_loaders import WikipediaLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import SKLearnVectorStore


# ## 1. Carga de documentos

# In[ ]:


loader = WikipediaLoader(query='Lenguaje Python',lang="es")
documents = loader.load()


# In[ ]:


documents


# In[ ]:


len(documents)


# ## 2. Split de Documentos

# In[ ]:


# División en fragmentos
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)


# In[ ]:


len(docs)


# ## 3. Conectar a OpenAI para los embeddings

# In[ ]:


f = open('../OpenAI_key.txt')
api_key = f.read()
funcion_embedding = OpenAIEmbeddings(openai_api_key=api_key)


# ## 4. Incrustar documentos en BD Vectores

# In[ ]:


persist_path="./ejemplo_wiki_bd"  #ruta donde se guardará la BBDD vectorizada

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


# ## 5a. Consulta normal similitud coseno

# In[ ]:


#Creamos un nuevo documento que será nuestra "consulta" para buscar el de mayor similitud en nuestra Base de Datos de Vectores y devolverlo
consulta = "¿Por qué el lenguaje Python se llama así?"
docs = vector_store.similarity_search(consulta)
print(docs[0].page_content)


# ## 5b. Consulta con compresión contextual usando LLMs

# In[ ]:


from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


# In[ ]:


llm = ChatOpenAI(temperature=0,openai_api_key=api_key) #el parámetro temperatura define la aleatoriedad de las respuestas, temperatura = 0 significa el mínimo de aleatoriedad
compressor = LLMChainExtractor.from_llm(llm)


# In[ ]:


compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vector_store.as_retriever())


# In[ ]:


compressed_docs = compression_retriever.invoke("¿Por qué el lenguaje Python se llama así?")


# In[ ]:


compressed_docs[0].page_content

