#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import SimpleSequentialChain, LLMChain, TransformChain
from langchain_community.document_loaders import WikipediaLoader

load_dotenv()
llm = ChatOpenAI()

consulta_wikipedia = input('Introduce el tema sobre el que quieres buscar información en Wikipedia: ')
idioma_final = input('Introduce el idioma al que quieres traducir el texto: ')

loader = WikipediaLoader(query=consulta_wikipedia, lang="es", load_max_docs=10)
data = loader.load()

print('\nWikepedia data:')
print(data[0].page_content)

texto_entrada = data[0].page_content


def transformer_function(inputs: dict) -> dict:
    texto = inputs['texto']
    primer_parrafo = texto.split('\n')[0]
    return {'salida': primer_parrafo}


transform_chain = TransformChain(input_variables=['texto'], output_variables=['salida'], transform=transformer_function)

template1 = "Crea un resumen en una línea del siguiente texto:\n{texto}"
prompt = ChatPromptTemplate.from_template(template1)
summary_chain = LLMChain(llm=llm, prompt=prompt, output_key="texto_resumen")

template2 = "Traduce a" + idioma_final + "el siguiente texto:\n{texto}"
prompt = ChatPromptTemplate.from_template(template2)
translate_chain = LLMChain(llm=llm, prompt=prompt, output_key="texto_traducido")

sequential_chain = SimpleSequentialChain(
    chains=[transform_chain, summary_chain, translate_chain],
    verbose=True)

result = sequential_chain(texto_entrada)

print('\nResultado:')
print(result)