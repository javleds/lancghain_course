#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader

load_dotenv()
chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

texto = "Esto es un texto enviado a OpenAI para ser incrustado en un vector n-dimensional"
embedded_text = embeddings.embed_query(texto)

print('\n Type of embedded_text:')
print(type(embedded_text))

print('\n Embedded text:')
print(embedded_text)

loader = CSVLoader('data_sources/datos_ventas_small.csv', csv_args={'delimiter': ';'})
data = loader.load()
print('\n Type of CSV data:')
print(type(data))
print('\n data at 0:')
print(type(data[0]))

# No podemos incrustar el objeto "data" puesto que es una lista de documentos, lo que espera es una string
# embedded_docs = embeddings.embed_documents(data) Creamos una comprensión de listas concatenando el campo
# "page_content" de todos los documentos existentes en la lista "data"
# [elemento.page_content for elemento in data]

embedded_docs = embeddings.embed_documents([elemento.page_content for elemento in data])
# Verificamos cuántos vectores ha creado (1 por cada registro del fichero CSV con datos)
print('\n Length of embedded docs:')
print(len(embedded_docs))

# Vemos un ejemplo del vector creado para el primer registro
print('\n Example of first element of the embedded docs:')
print(embedded_docs[1])
