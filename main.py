from typing import Set
from backend.core import run_llm

import streamlit as st

st.header("Company Test Bot")
prompt = st.chat_input("Prompt")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_lit = list(source_urls)
    sources_lit.sort()
    sources_string ="sources:\n"
    for i, source in enumerate(sources_lit):
        sources_string += f"{i+1}. {source}\n"

    return sources_string


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt)
        formatted_response = (
            f"{generated_response['answer']} \n\n"
        )
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response['answer']))

if st.session_state['chat_answers_history']:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        with st.chat_message("user"):
            st.write(user_query)
        with st.chat_message("assistant"):
            st.write(generated_response)

        print(user_query, 'query')
        print(generated_response, 'response')

# step1 : load pdf => step2 cohere embedding => stepn: Question answer
