#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías e instancia de modelo de chat

# In[ ]:


from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
f = open('../OpenAI_key.txt')
api_key = f.read()
chat = ChatOpenAI(openai_api_key=api_key)


# ## Carga datos CSV

# In[ ]:


from langchain.document_loaders import CSVLoader #pip install langchain-community en una terminal


# In[ ]:


#Cargamos el fichero CSV
loader = CSVLoader('data_sources/datos_ventas_small.csv',csv_args={'delimiter': ';'})


# In[ ]:


#Creamos el objeto "data" con los datos desde el cargador "loader"
data = loader.load()


# In[ ]:


#print(data) #Vemos que se ha creado un documento por cada fila donde el campo page_content contiene los datos


# In[ ]:


type(data)


# In[ ]:


data[0]


# In[ ]:


data[1]


# In[ ]:


print(data[1].page_content)


# ## Carga datos HTML

# In[ ]:


from langchain.document_loaders import BSHTMLLoader #pip install beautifulsoup4 en una terminal


# In[ ]:


loader = BSHTMLLoader('data_sources/ejemplo_web.html')


# In[ ]:


data = loader.load()


# In[ ]:


data


# In[ ]:


print(data[0].page_content)


# ## Carga datos PDF

# In[ ]:


from langchain.document_loaders import PyPDFLoader #pip install pypdf en una terminal


# In[ ]:


loader = PyPDFLoader('data_sources/DocumentoTecnologiasEmergentes.pdf')


# In[ ]:


pages = loader.load_and_split()


# In[ ]:


type(pages)


# In[ ]:


pages[0]


# In[ ]:


print(pages[0].page_content)


# In[ ]:





# # Caso de uso: Resumir PDFs

# In[ ]:


contenido_pdf=pages[0].page_content


# In[ ]:


contenido_pdf


# In[ ]:


human_template = '"Necesito que hagas un resumen del siguiente texto: \n{contenido}"'
human_prompt = HumanMessagePromptTemplate.from_template(human_template)


# In[ ]:


chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

chat_prompt.format_prompt(contenido=contenido_pdf)


# In[ ]:


solicitud_completa = chat_prompt.format_prompt(contenido=contenido_pdf).to_messages()


# In[ ]:


result = chat.invoke(solicitud_completa)


# In[ ]:


result.content


# In[ ]:


#Resumir el documento completo
#Creamos una string concatenando el contenido de todas las páginas
documento_completo = ""
for page in pages:
    documento_completo += page.page_content  # Supongamos que cada página tiene un atributo 'text'

print(documento_completo)


# In[ ]:


solicitud_completa = chat_prompt.format_prompt(contenido=documento_completo).to_messages()


# In[ ]:


result = chat.invoke(solicitud_completa)


# In[ ]:


result.content


# In[ ]:




