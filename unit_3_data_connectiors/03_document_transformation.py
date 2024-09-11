#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# #  Carga del fichero

# In[ ]:


with open('data_sources/HistoriaEspana.txt', encoding="utf8") as file:
    texto_completo = file.read()


# In[ ]:


# Números de caracteres
len(texto_completo)


# In[ ]:


# Número de palabras
len(texto_completo.split())


# ## Transformador "Character Text Splitter"

# In[ ]:


from langchain.text_splitter import CharacterTextSplitter


# In[ ]:


text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000) #Indicamos que divida cuando se encuentra 1 salto de línea y trate de hacer fragmentos de 1000 caracteres


# In[ ]:


texts = text_splitter.create_documents([texto_completo]) #Creamos documentos gracias al transformador


# In[ ]:


print(type(texts)) #Verificamos el tipo del objeto obtenido
print('\n')
print(type(texts[0])) #Verificamos el tipo de cada elemento
print('\n')
print(texts[0])


# In[ ]:


len(texts[0].page_content)


# In[ ]:


texts[1]


# In[ ]:


len(texts[1].page_content)

