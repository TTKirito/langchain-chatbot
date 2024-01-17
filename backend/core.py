from typing import Any, List, Dict
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.pgvector import PGVector

COLLECTION_NAME = 'state_of_union_vectors'

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="mysecretpassword",
)

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    store = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )

    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=store.as_retriever(),
        return_source_documents=True
    )
    # 7 Now, if we want to ask a question, take that as a query, embed it into a vector,
    # And now we can calculate the closest vectors saved to the query vector that we embedded.
    return qa({ "question": query, "chat_history": chat_history })

if __name__== '__main__':
    print(run_llm(query="what is three types of Innovation Capital"))
