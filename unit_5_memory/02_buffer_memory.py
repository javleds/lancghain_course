#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías iniciales e instancia de modelo de chat

# In[ ]:


from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
f = open('../OpenAI_key.txt')
api_key = f.read()
llm = ChatOpenAI(openai_api_key=api_key)


# #  Crear objeto ConversationBufferMemory

# In[ ]:


memory = ConversationBufferMemory()


# ## Conectar una conversación a la memoria

# In[ ]:


#Creamos una instancia de la cadena conversacional con el LLM y el objeto de memoria
conversation = ConversationChain(llm=llm,memory = memory,verbose=True)


# In[ ]:


#Ejemplo con RunnableWithMessageHistory: https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html


# In[ ]:


#Lanzamos el primer prompt (human message)
conversation.predict(input="Hola, necesito saber cómo usar mis datos históricos para crear un bot de preguntas y respuestas")


# In[ ]:


#Lanzamos el segundo prompt (human message)
conversation.predict(input="Necesito más detalle de cómo implementarlo")


# In[ ]:


#Obtenemos el histórico
print(memory.buffer)


# In[ ]:


#Cargamos la variable de memoria
memory.load_memory_variables({})


# ## Guardar y Cargar la memoria (posterior uso)

# In[ ]:


conversation.memory


# In[ ]:


import pickle
pickled_str = pickle.dumps(conversation.memory) #Crea un objeto binario con todo el objeto de la memoria


# In[ ]:


#pickled_str #objeto binario que guarda cualquier tipo de información


# In[ ]:


with open('memory.pkl','wb') as f: #wb para indicar que escriba un objeto binario, en este caso en la misma ruta que el script
    f.write(pickled_str)


# In[ ]:


memoria_cargada = open('memory.pkl','rb').read() #rb para indicar que leemos el objeto binario


# In[ ]:


llm = ChatOpenAI(openai_api_key=api_key) #Creamos una nueva instancia de LLM para asegurar que está totalmente limpia
conversacion_recargada = ConversationChain(
    llm=llm, 
    memory = pickle.loads(memoria_cargada),
    verbose=True
)


# In[ ]:


conversacion_recargada.memory.buffer


# In[ ]:




