#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore

load_dotenv()

embedding_function = OpenAIEmbeddings()
persist_path = "./ejemplo_wiki_bd"

if Path('ejemplo_wiki_bd').exists():
    print('\nLoading local vector store')
    vector_store = SKLearnVectorStore(
        embedding=embedding_function, persist_path=persist_path, serializer="parquet"
    )
else:
    print('\nCreating new local vector store')
    loader = WikipediaLoader(query='Lenguaje Python', lang="es")
    documents = loader.load()

    # División en fragmentos
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
    docs = text_splitter.split_documents(documents)

    # Creamos la DB de vectores a partir de los documentos y la función embeddings
    vector_store = SKLearnVectorStore.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_path=persist_path,
        serializer="parquet",  # el serializador o formato de la BD lo definimos como parquet
    )

    vector_store.persist()

# 5a. Consulta normal similitud coseno Creamos un nuevo documento que será nuestra "consulta" para buscar el de
# mayor similitud en nuestra Base de Datos de Vectores y devolverlo
consulta = "¿Por qué el lenguaje Python se llama así?"
docs = vector_store.similarity_search(consulta)
print('\n5a. Consulta local:')
print(docs[0].page_content)

# 5b. Consulta con compresión contextual usando LLMs
llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                       base_retriever=vector_store.as_retriever())

compressed_docs = compression_retriever.invoke("¿Por qué el lenguaje Python se llama así?")
print('\n5b. Complemento con OpenAI:')
print(compressed_docs[0].page_content)
