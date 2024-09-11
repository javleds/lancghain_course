#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.chains.question_answering import load_qa_chain

# from langchain.chains.qa_with_sources import load_qa_with_sources_chain #Opción que proporciona también la fuente
# de datos de la respuesta

load_dotenv()
llm = ChatOpenAI()
embedding_function = OpenAIEmbeddings()
vector_store_connection = SKLearnVectorStore(
    embedding=embedding_function,
    persist_path="../unit_3_data_connectiors/ejemplosk_embedding_db",
    serializer="parquet")

# chain_type='stuff' se usa cuando se desea una manera simple y directa de cargar
# y procesar el contenido completo sin dividirlo en fragmentos más pequeños.
# Es ideal para situaciones donde el volumen de datos no es demasiado grande
# y se puede manejar de manera eficiente por el modelo de lenguaje en una sola operación.
chain = load_qa_chain(llm, chain_type='stuff')
question = "¿Qué pasó en el siglo de Oro?"
docs = vector_store_connection.similarity_search(question)
print('\nRespuesta con invoke:')
print(chain.invoke({
    "input_documents": docs,
    "question": question
})["output_text"])
