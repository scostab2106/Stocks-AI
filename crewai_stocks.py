#import libs

import json
import os
from datetime import datetime

import yfinance as yf

from crewai import  Agent, Task, crew, process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

#CRIANDO YAHOO FINANCE TOOOLS

def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-20", end="2024-08-19")
    return(stock) 

yft = Tool(
name = "Yahoo Finance Tools",
description =" ",
func = lambda ticket: fetch_stock_price(ticket) 

)


#IMPORTANDO OPENAI

os.environ['OPEN_API_KEY'] = "",
llm = ChatOpenAI(model = "gpt-3.5-turbo") 

stockPriceAnalyst = Agent(

role ="",
goal="",
backstore = "",
verbose = True,
llm = llm,
max_iter = 5,
memory = True,
tools = [],
allow_delegation = False

)

get_StockPrice = Task(

description = "",
expected_output = "",

agent = stockPriceAnalyst

)

#IMPORTANDO A TOOL DE SEARCH 

search_tool = DuckDuckGoSearchResults(backend='news',num_results = 10 )

stockNewsAnalyst = Agent(

role ="",
goal="",
backstore = "",
verbose = True,
llm = llm,
max_iter = 5,
memory = True,
tools = [search_tool],
allow_delegation = False

)

get_news = Task(

description = "",
expected_output = "",

agent =  stockNewsAnalyst


)

stockAnalystWriter = Agent(

role ="",
goal="",
backstore = "",
verbose = True,
llm = llm,
max_iter = 5,
memory = True,
tools = [],
allow_delegation = True   
)

get_writeAnalyses = Task(

description = " ",
expected_output = " ", 

agent = stockAnalystWriter,
context = [get_StockPrice, get_news]

)

crew = crew(

    agents = [stockPriceAnalyst,stockNewsAnalyst,stockAnalystWriter],
    tasks = [get_StockPrice,get_news,get_writeAnalyses],
    verbose = 2,
    process = process.hierarchical,
    full_output = True,
    share_crew = False,
    manager_llm = llm,
    max_iter = 15

)



results = crew.kickoff(inputs= {'ticket': 'AAPL'})

list(results.keys())

results['final_output']


with st.sidebar:
    st.header('Enter the stock to research')

    with st.form(key='research_form' ):
        topic  = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label = "Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else: 
        results = crew.kickoff(inputs= {'ticket': topic})

        st.subheader("Results of your research:")
        st.write(results['final_output'])

        
    