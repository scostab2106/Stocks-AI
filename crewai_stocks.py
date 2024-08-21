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
description ="Fetches stocks prices for {ticket} from the last year about a specific company from Yahoo Finance API",
func = lambda ticket: fetch_stock_price(ticket) 

)


#IMPORTANDO OPENAI

os.environ['OPEN_API_KEY'] = "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",
llm = ChatOpenAI(model = "gpt-3.5-turbo") 

stockPriceAnalyst = Agent(

role ="Senior stock price Analyst",
goal="Find the {ticket} stock price and analyses trends",
backstore = "You're a highly experienced in analyzing the price of an specific stock and make predictions about its future price",
verbose = True,
llm = llm,
max_iter = 5,
memory = True,
tools = [],
allow_delegation = False

)

get_StockPrice = Task(

description = "Analyze the stock {ticket} price history and create a trend analyses of up, down or sideways",
expected_output = "Specify the current trend stock price - up, down or sideways. eg. stock = 'APPL, price UP in 2 days",

agent = stockPriceAnalyst

)

#IMPORTANDO A TOOL DE SEARCH 

search_tool = DuckDuckGoSearchResults(backend='news',num_results = 10 )

stockNewsAnalyst = Agent(

role ="Stock News Analys",
goal="Create a short summary of the market news related to the stock {ticket} company. Specify a the current trend - up, down or sideway with the news context. For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.",
backstore = "You're highly experienced in analyzing the market trends and news and have tracked assest for more then 10 years. You're also master level analyst in the tradicional markets and have deep understanding of human psychology. You understand news, theirs tittles and information, but you look at those with a health dose of skepticism. You consider also the source of the news articles",
verbose = True,
llm = llm,
max_iter = 5,
memory = True,
tools = [search_tool],
allow_delegation = False

)

get_news = Task(

description = f"Take the stock and always include BTC to it (if not request). Use the search tool to search each one individually. The current date is {datetime.now()}. Compose the results into a helpfull report",
expected_output = """A summary of the overall market and one sentence summary for each request asset. Include a fear/greed score for each asset based on the news. Use format:  "
<STOCK ASSET>
<SUMMARY BASED ON NEWS>
<TREND PREDICTION>
<FEAR/GREED SCORE>
""" ,
agent =  stockNewsAnalyst


)

stockAnalystWriter = Agent(

role ="Senior stock analyst writer",
goal="Analyze the trends price and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock reporta and price trend.",
backstore = "You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories and narratives that resonate witch wider audiences. You understand macro factores and combine multiple theories - eg. cycle theory and fundamental analyses. You're able to hold multiple opinions when analyzing anything.",
verbose = True,
llm = llm,
max_iter = 5,
allow_delegation = True   
)

get_writeAnalyses = Task(

description = "Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company that is brief and highlights the most important points. Focus on the stock price trend, news and fear/greed score. What are the near future considerations? Include the precious analyses of stock trend and news summary.",
expected_output = """An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. It should contain:  
- 3 bullets executive summary
- Introduction - set the overall picture and spike up the interest
- Main part provides the meat of the analysis includinhg the news summary and fear/greed scores
- Summary - key facts and concrete future trend prediction - up, down or sideways.
""",
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

        
    