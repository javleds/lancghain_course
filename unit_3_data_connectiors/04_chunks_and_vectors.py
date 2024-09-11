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


# #  Incrustación de texto (embedding)

# In[ ]:


from langchain_openai import OpenAIEmbeddings


# In[ ]:


embeddings = OpenAIEmbeddings(openai_api_key=api_key)


# In[ ]:


texto = "Esto es un texto enviado a OpenAI para ser incrustado en un vector n-dimensional"


# In[ ]:


embedded_text = embeddings.embed_query(texto)


# In[ ]:


type(embedded_text)


# In[ ]:


embedded_text


# ## Incrustación de documentos

# In[ ]:


from langchain.document_loaders import CSVLoader


# In[ ]:


loader = CSVLoader('Fuentes datos/datos_ventas_small.csv',csv_args={'delimiter': ';'})


# In[ ]:


data = loader.load()


# In[ ]:


type(data)


# In[ ]:


type(data[0])


# In[ ]:


#No podemos incrustar el objeto "data" puesto que es una lista de documentos, lo que espera es una string
#embedded_docs = embeddings.embed_documents(data)


# In[ ]:


#Creamos una comprensión de listas concatenando el campo "page_content" de todos los documentos existentes en la lista "data"
[elemento.page_content for elemento in data]


# In[ ]:


embedded_docs = embeddings.embed_documents([elemento.page_content for elemento in data])


# In[ ]:


#Verificamos cuántos vectores a creado (1 por cada registro del fichero CSV con datos)
len(embedded_docs)


# In[ ]:


#Vemos un ejemplo del vector creado para el primer registro
embedded_docs[1]


# In[ ]:




