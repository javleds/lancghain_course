#!/usr/bin/env python
# coding: utf-8

# pip install langchain-community, pypdf y lxml en una terminal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders import CSVLoader, BSHTMLLoader, PyPDFLoader

load_dotenv()
chat = ChatOpenAI()

# Cargamos el fichero CSV
loader = CSVLoader('data_sources/datos_ventas_small.csv', csv_args={'delimiter': ';'})
data = loader.load()
print('CSV Loader:')
print(data[1].page_content)

# ## Carga datos HTML
loader = BSHTMLLoader('data_sources/ejemplo_web.html')
data = loader.load()
print('HTML Loader:')
print(data[0].page_content)

# ## Carga datos PDF
loader = PyPDFLoader('data_sources/DocumentoTecnologiasEmergentes.pdf')
pages = loader.load_and_split()
print('PDF Loader:')
print(pages[0].page_content)

# # Caso de uso: Resumir PDFs
contenido_pdf = pages[0].page_content
human_template = '"Necesito que hagas un resumen del siguiente texto: \n{contenido}"'
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
chat_prompt.format_prompt(contenido=contenido_pdf)

solicitud_completa = chat_prompt.format_prompt(contenido=contenido_pdf).to_messages()
result = chat.invoke(solicitud_completa)
print('PDF answer 1:')
print(result.content)

# Resumir el documento completo
# Creamos una string concatenando el contenido de todas las páginas
documento_completo = ""
for page in pages:
    documento_completo += page.page_content  # Supongamos que cada página tiene un atributo 'text'

print('PDF completo:')
print(documento_completo)

solicitud_completa = chat_prompt.format_prompt(contenido=documento_completo).to_messages()
result = chat.invoke(solicitud_completa)

print('PDF answer 2:')
print(result.content)
