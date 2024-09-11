#!/usr/bin/env python
# coding: utf-8

from langchain.text_splitter import CharacterTextSplitter

with open('data_sources/HistoriaEspana.txt', encoding="utf8") as file:
    texto_completo = file.read()

print('\nNúmero de caracteres:')
print(len(texto_completo))

print('\nNúmero de palabras:')
print(len(texto_completo.split()))

# ## Transformador "Character Text Splitter"

# Indicamos que divida cuando se encuentra 1 salto de línea y trate de hacer fragmentos de 1000 caracteres
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000)
texts = text_splitter.create_documents([texto_completo])  # Creamos documentos gracias al transformador

print(type(texts))  # Verificamos el tipo del objeto obtenido
print('\n')
print(type(texts[0]))  # Verificamos el tipo de cada elemento
print('\n')
print(texts[0])

print('\n Page content:')
print(len(texts[0].page_content))
