from typing import Union

import numpy as np
from advertools import crawl
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from pandas import Series, DataFrame
from langchain.vectorstores.pgvector import PGVector

COLLECTION_NAME = 'state_of_union_vectors'

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="mysecretpassword",
)
def craw(site: str) -> Union[Series, DataFrame]:
    #1: craw website return dataframe
    crawl(site, 'simp.jl', follow_links=True)
    crawlDf = pd.read_json('simp.jl', lines=True)
    crawlDf = crawlDf[['body_text', 'header_links_text', 'og:title', 'h1', 'h2', 'h3', 'h4', 'h5', 'title']]
    crawlDf.head()
    crawlDf.replace(np.nan, 0)
    crawlDf = crawlDf.where(~crawlDf.isna(), 0)
    crawlDf = crawlDf.fillna(0)
    return crawlDf


def embeddingCraw(crawlDf) -> None:
    total_length = len(crawlDf)
    batch_size = 10
    #2: Split dataFrame into many parts because of limited tokens when embedding // <1000 token
    for batch_start in range(0, total_length, batch_size):
        batch_end = min(batch_start + batch_size, total_length)
        df1 = crawlDf.iloc[batch_start:batch_end, :]
        #3: Read documents using langchain document loader
        loader = DataFrameLoader(df1, page_content_column="body_text")
        docs = loader.load()
        print(len(docs))
        #4: Take large text and divide it into parts
        # text_splitterPDF = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10, separator="\n")
        text_splitterPDF = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        textsPDF = text_splitterPDF.split_documents(docs)
        #5: Take all these chunks and embed them using an embedding model and turn them into a vector that each vector represents that chunk.
        # Each vector will be a list of numbers representing the given segment that embeds.
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        #6: Save them to a vector database
        PGVector.from_documents(
            embedding=embeddings,
            documents=textsPDF,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING
        )

        print(f"Inserted {batch_end}/{total_length} chunks")


# Request too large for text-embedding-ada-002
if __name__ == '__main__':
    crawl_df = craw('https://vinova.sg')
    embeddingCraw(crawl_df)


# ---------------------------------NOTE--------------------------------------
# pipenv install git+https://github.com/julian-r/python-magic.git
# pipenv install unstructured - q
# pipenv install unstructured[local - inference] - q
# pipenv install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2 -q
# sudo apt-get install poppler-utils
# pipenv install tiktoken -q
# pipenv install pytesseract
# sudo apt install tesseract-ocr
# pipenv install advertools
# docker run --name pgvector-demo -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d ankane/pgvector
# CREATE DATABASE vector_db;
# CREATE EXTENSION IF NOT EXISTS vector