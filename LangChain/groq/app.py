import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
load_dotenv()

# Load GROQ API key
groq_api_key = os.environ["GROQ_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://apexlegends.fandom.com/wiki/Apex_Legends_Wiki")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectordb = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)

st.title("Apex Legends Info")
st.subheader("will answer your question based on data on this page \"https://apexlegends.fandom.com/wiki/Apex_Legends_Wiki\"")
model = ChatGroq(groq_api_key = groq_api_key,
                 model="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only. Please provide the most accurate response based on the question.
    <context>{context}</context>
    Question:{input}
    """
)

document_chain = create_stuff_documents_chain(model, prompt)
retriever = st.session_state.vectordb.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Ask here")

if prompt :
    start = time.process_time()
    response = retriever_chain.invoke({"input":prompt})
    print("Response Time : ", time.process_time() - start)
    st.write(response['answer'])

    # with st.expander("See explanation"):
    #     for i, doc in enumerate(response['context']):
    #         st.write(doc.page_content)