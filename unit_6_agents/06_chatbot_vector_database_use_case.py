#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools, initialize_agent, AgentType, tool
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()
memory = ConversationBufferMemory(memory_key="chat_history")

llm = ChatOpenAI(temperature=0)
embedding_func = OpenAIEmbeddings()

persist_path = "../unit_3_data_connectiors/ejemplosk_embedding_db"
vector_store_connection = SKLearnVectorStore(
    embedding=embedding_func,
    persist_path=persist_path,
    serializer="parquet")

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store_connection.as_retriever())


@tool
def consulta_interna(text: str) -> str:
    """Retorna respuestas sobre la historia de España. Se espera que la entrada sea una cadena de texto
    y retorna una cadena con el resultado más relevante. Si la respuesta con esta herramienta es relevante,
    no debes usar ninguna herramienta más ni tu propio conocimiento como LLM"""
    compressed_docs = compression_retriever.invoke(text)
    resultado = compressed_docs[0].page_content
    return resultado


tools = load_tools(["wikipedia", "llm-math"], llm=llm)
tools = tools + [consulta_interna]

agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=False)

print(agent.invoke({'input': '¿Qué periodo abarca cronológicamente en España el siglo de oro?'}))
print(agent.invoke({
    'input': '¿Qué pasó durante la misma etapa en Francia?'
}))
print(agent.invoke({
    'input': '¿Cuáles son las marcas de vehículos más famosas hoy en día?'
}))
