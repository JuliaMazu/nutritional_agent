#from dotenv import load_dotenv
#load_dotenv()
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from operator import itemgetter
import streamlit as st


def load_content(uploaded_file):
    """create list of pages from the loaded document"""
    extention = uploaded_file.split(".")[-1]
   #st.write(uploaded_file)
    if extention == "pdf":
        loader = PyPDFLoader(uploaded_file)
    elif extention == "docx":
        loader = Docx2txtLoader(uploaded_file)
    elif extention == "txt":
        loader = TextLoader(uploaded_file)
    else:
        return st.write("Provide one the followig format: pdf, txt, docx")
    return loader.load()

def concat_pages(loaded, range):
    """create total txt file based on user chosen range"""
    min, max = range[0]-1, range[1]-1
    pages = ''
    for page in loaded[min:max]:
        pages += page.page_content
    return pages


def splitter(pages):
    """to create chunks of desired lenth based on value_chunk"""
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=value_chunk, chunk_overlap=value_chunk/20)

    sample_chunks = recursive_splitter.split_text(pages)
    chunks_as_documents = [Document(page_content=chunk) for chunk in sample_chunks]
    return chunks_as_documents

def embeddings(api_key):
    """vectorize chunks save at temporary db"""
    chunks_as_documents = splitter(pages)
    #print("!!!!!")
    #print(os.getenv("HF_TOKEN"))
    mistral_embeddings = MistralAIEmbeddings(api_key=api_key, model="mistral-embed")
    
    vector_db = Chroma.from_documents(chunks_as_documents, mistral_embeddings)
    retriever = vector_db.as_retriever()
    #print(retriever.invoke("best time to eat bananas"))
    return retriever


               ### ---- Variable for streamlit and chat components  ---###

template = """You are the sport nutrition guide.
        Answer the question using only the following context. Do not rely on external sources:
        {context}
        Make a summary from the context. Do not mention document id for user
        Use {history} to get additional and personalized information for your answer
        Question : {question}
        """
   
prompt = ChatPromptTemplate(
            messages=[
                    (
                        "system", template
                    ),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),
                    ]
                            )
loaded, pages, api_key, values_pages, chain, retriever = None, None, None, None, None, None


                         #### ----- PAGE ------ ####
                           ## Side bar  ##
with st.sidebar:
    uploaded_files = st.file_uploader(
    "Choose a file to analyse", accept_multiple_files=True
    )
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        # st.write("filename:", uploaded_file.name)
        loaded = load_content(uploaded_file.name) #create list of pages from file

    #st.subheader("Chunk choise")
    value_chunk = st.slider("Chunk size of the text", 500, 5000, value=1000, step=400)
    st.write("Chosen chunk size:", value_chunk)

    #st.subheader("Page range selection")
    if loaded:
        values_pages = st.slider("Which page to analyse", 1, len(loaded)+1, (1, len(loaded)+1))
        st.write("Selected pages:", values_pages)
    else:
        st.write("Please upload the file")

    if loaded and values_pages:
        pages = concat_pages(loaded, values_pages) #based on chosen value create total file
    
    if api_key is None: 
        api_key = st.text_input(
            "Provide your Mistral API 👇",
            label_visibility="visible", type="password")
        

#create retriever for chat completion if there is a loaded document and an api key
if api_key and pages:
    retriever = embeddings(api_key)


                    #### MAIN Page layout and chat deffinition ####

st.title("Simple text analyser for your sport nutrition data")


#update memory 
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

memory = st.session_state.memory
#print(memory.buffer)

if api_key and retriever:
                                  ### Chat completion ###
    llm = ChatMistralAI(
        model="mistral-small-latest", temperature=0.3, api_key=api_key)
    
    # Créer la chaîne RAG
    chain_rag = (
        RunnablePassthrough.assign(context=itemgetter("question") | retriever)
        | prompt
        | llm
        | StrOutputParser())
    #chain with memory
    chain = (
        RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
            )
        | RunnablePassthrough.assign(
            answer=chain_rag
            )
        | RunnableLambda(
          lambda x: memory.save_context({"question": x["question"]}, {"output": x["answer"]}) or x["answer"]
            )
        )
    
if chain and retriever:
    if "messages" not in st.session_state:
        st.session_state.messages = [] # Stores chat history

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if prompt := st.chat_input("What do you want to know about this file"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                response = chain.invoke({"question" : prompt})
                st.markdown(response)
                st.session_state.messages.append({"role":"assistant", "content":response})

