#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## 0.Importar librerías iniciales e instancia de modelo de chat

# In[ ]:


from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.agents import load_tools,initialize_agent,AgentType,create_react_agent,AgentExecutor
f = open('../OpenAI_key.txt')
api_key = f.read()
llm = ChatOpenAI(openai_api_key=api_key,temperature=0) #Recomendable temperatura a 0 para que el LLM no sea muy creativo, vamos a tener muchas herramientas a nuestra disposición y queremos que sea más determinista


# ## 1.Cargamos la BD Vectores y el compresor

# In[ ]:


#Podríamos establecer que tuviera memoria
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history") #ponemos una denominada clave a la memoria "chat_history"


# In[ ]:


from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

funcion_embedding = OpenAIEmbeddings(openai_api_key=api_key)
persist_path="../Bloque 3_Conectores de Datos/ejemplosk_embedding_db"
vector_store_connection = SKLearnVectorStore(embedding=funcion_embedding, persist_path=persist_path, serializer="parquet")
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vector_store_connection.as_retriever())


# ## 2.Creamos una nueva herramienta a partir de la BD Vectorial para obtener resultados optimizados

# In[ ]:


from langchain.agents import tool


# In[ ]:


@tool
def consulta_interna(text: str) -> str:
    '''Retorna respuestas sobre la historia de España. Se espera que la entrada sea una cadena de texto
    y retorna una cadena con el resultado más relevante. Si la respuesta con esta herramienta es relevante,
    no debes usar ninguna herramienta más ni tu propio conocimiento como LLM'''
    compressed_docs = compression_retriever.invoke(text)
    resultado = compressed_docs[0].page_content
    return resultado


# In[ ]:


tools = load_tools(["wikipedia","llm-math"],llm=llm)


# In[ ]:


tools=tools+[consulta_interna]


# ## 3.Creamos el agente y lo ejecutamos 

# In[ ]:


agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,memory=memory,verbose=True)


# In[ ]:


agent.invoke("¿Qué periodo abarca cronológicamente en España el siglo de oro?")


# In[ ]:


agent.invoke("¿Qué pasó durante la misma etapa en Francia?") #Gracias a tener memoria compara en esas fechas qué ocurrió en Francia


# In[ ]:


agent.invoke("¿Cuáles son las marcas de vehículos más famosas hoy en día?") #Pregunta que no podemos responder con nuestra BD Vectorial

