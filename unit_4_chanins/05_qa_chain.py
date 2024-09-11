#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías iniciales e instancia de modelo de chat

# In[ ]:


from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import SimpleSequentialChain, LLMChain,TransformChain
f = open('../OpenAI_key.txt')
api_key = f.read()
llm = ChatOpenAI(openai_api_key=api_key)


# ### Conectar a BD Vectores

# In[ ]:


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore


# In[ ]:


embedding_function = OpenAIEmbeddings(openai_api_key=api_key)


# In[ ]:


vector_store_connection = SKLearnVectorStore(embedding=embedding_function, persist_path="../Bloque 3_Conectores de Datos/ejemplosk_embedding_db", serializer="parquet")


# ## Cargar cadena QA

# In[ ]:


from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain #Opción que proporciona también la fuente de datos de la respuesta


# In[ ]:


chain = load_qa_chain(llm,chain_type='stuff') #chain_type='stuff' se usa cuando se desea una manera simple y directa de cargar y procesar el contenido completo sin dividirlo en fragmentos más pequeños. Es ideal para situaciones donde el volumen de datos no es demasiado grande y se puede manejar de manera eficiente por el modelo de lenguaje en una sola operación.


# In[ ]:


question = "¿Qué pasó en el siglo de Oro?"


# In[ ]:


docs = vector_store_connection.similarity_search(question)


# In[ ]:


chain.run(input_documents=docs,question=question)


# ### Alternativa con método invoke

# In[ ]:


#Estructurar un diccionario con los parámetros de entrada
inputs = {
    "input_documents": docs,
    "question": question
}


# In[ ]:


chain.invoke(inputs)["output_text"]

