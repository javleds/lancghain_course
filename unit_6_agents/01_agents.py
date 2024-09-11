#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías iniciales e instancia de modelo de chat

# In[ ]:


from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
f = open('../OpenAI_key.txt')
api_key = f.read()
llm = ChatOpenAI(openai_api_key=api_key,temperature=0) #Recomendable temperatura a 0 para que el LLM no sea muy creativo, vamos a tener muchas herramientas a nuestra disposición y queremos que sea más determinista


# In[ ]:


from langchain.agents import load_tools,initialize_agent,AgentType,create_react_agent,AgentExecutor


# ## Definimos las herramientas a las que tendrá acceso el agente (aparte del propio motor LLM)

# In[ ]:


tools = load_tools(["llm-math",],llm=llm) #Lista de herramientas disponibles: https://python.langchain.com/v0.1/docs/integrations/tools/


# ## Inicializamos y ejecutamos el Agente

# In[ ]:


#dir(AgentType) #Vemos los diferentes tipos de agente a usar


# In[ ]:


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,handle_parsing_errors=True) #Usamos el Zero Shot porque no estamos dando ningún ejemplo, solo pidiendo al agente hacer una tarea sin ejemplos previos


# In[ ]:


agent.run("Dime cuánto es 1598 multiplicado por 1983 y después sumas 1000")


# ## Alternativa agente con create_react_agent (indicaciones explícitas)

# In[ ]:


template = '''Responde lo mejor que puedas usando tu conocimiento como LLM o bien las siguientes herramientas:
{tools}
Utiliza el siguiente formato:
Pregunta: la pregunta de entrada que debes responder
Pensamiento: siempre debes pensar en qué hacer
Acción: la acción a realizar debe ser una de [{tool_names}]
Entrada de acción: la entrada a la acción.
Observación: el resultado de la acción.
... (este Pensamiento/Acción/Introducción de Acción/Observación puede repetirse N veces,si no consigues el resultado tras 5 intentos, para la ejecución)
Pensamiento: ahora sé la respuesta final
Respuesta final: la respuesta final a la pregunta de entrada original
¡Comenzar! Recuerda que no siempre es necesario usar las herramientas
Pregunta: {input}
Pensamiento:{agent_scratchpad}'''

#agent_scratchpad: El agente no llama a una herramienta solo una vez para obtener la respuesta deseada, sino que tiene una estructura que llama a las herramientas repetidamente hasta obtener la respuesta deseada. Cada vez que llama a una herramienta, en este campo se almacena cómo fue la llamada anterior, información sobre la llamada anterior y el resultado.


# In[ ]:


prompt = PromptTemplate.from_template(template)


# In[ ]:


agente = create_react_agent(llm,tools,prompt)


# In[ ]:


agent_executor = AgentExecutor(
    agent=agente,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True
)


# In[ ]:


respuesta = agent_executor.invoke({"input": "Dime cuánto es 1598 multiplicado por 1983"})
print(respuesta)


# In[ ]:




