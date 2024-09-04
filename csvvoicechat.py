# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:16:24 2024

@author: yash
"""

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import uuid  # For generating unique file names
import pandas as pd
from gtts import gTTS
import tempfile
load_dotenv()

import streamlit as st
import speech_recognition as sr
from io import BytesIO

# Initialize recognizer class (for recognizing the speech)
recognizer = sr.Recognizer()

# Load environment variables
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

import pyttsx3

# Initialize the pyttsx3 engine
engine = pyttsx3.init()
# Set up Streamlit

st.set_page_config(page_title="DataSpeak 🤖 🎤 👨🏽‍💻: Turn your questions into conversations with data. ", page_icon="🤖")
st.title("Conversational RAG With CSV Uploads and Textual Voices 💻 🗣️")
#engine.say("Conversational RAG With CSV Uploads and Chat History")
#engine.runAndWait()
st.write("Upload CSV files and chat with their content")
#engine.say("Upload CSV files and chat with their content")
#engine.runAndWait()

llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-70b-8192')

# File uploader for CSV files
uploaded_files = st.file_uploader("Choose a CSV file", type="csv", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        # Generate a unique temporary file name for each upload
        temp_csv_path = f"./temp_{uuid.uuid4()}.csv"

        # Save the uploaded CSV file to the temporary path
        with open(temp_csv_path, "wb") as file:
            file.write(uploaded_file.getvalue())

        # Use the loader to load the CSV data
        loader = CSVLoader(file_path=temp_csv_path)
        data = loader.load()
        documents.extend(data)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=80 * 50, chunk_overlap=8 * 50)
        
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        system_prompt = (
             "You are an assistant for question-answering tasks. "
             "Use the following pieces of retrieved context to answer "
             "the question. If you don't know the answer, say that you "
             "don't know. Use three sentences maximum and keep the "
             "answer concise."
             "\n\n"
             "{context}"
             )
             
            

        prompt = ChatPromptTemplate.from_messages(
            
            [
                    ("system", system_prompt),
                    ("human", "{input}"),
            ]
            
            )
            
        question_answer_chain=create_stuff_documents_chain(llm,prompt)
        rag_chain=create_retrieval_chain(retriever,question_answer_chain)
            
        output_parser=StrOutputParser()
            
        chain=prompt|llm|output_parser
        text=""
        # Button to start the recording
        if st.button("Start Listening"):
            # Set up the microphone and adjust for ambient noise
            with sr.Microphone() as source:
                st.write("Listening...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)  # Listen for the first phrase and extract it into audio data

            # Transcribe the captured audio to text
            try:
                text = recognizer.recognize_google(audio)
                st.write("Transcribed Text:")
                st.text_area("Live Transcription", value=text, height=150)
            except sr.UnknownValueError:
                st.write("Google Speech Recognition could not understand the audio.")
            except sr.RequestError:
                st.write("Could not request results from Google Speech Recognition service.")
        
        
        
            
        
            
            
                
        context = retriever.get_relevant_documents(text)
        formatted_context = "\n".join([doc.page_content for doc in context])
        
            
            
    
    # Adjusting the invocation of the chain
        st.write(chain.invoke({"context": formatted_context, "input": text}))
        
        a=chain.invoke({"context": formatted_context, "input": text})
        with open("output.txt", "w") as file:
            
            
            file.write(a)
            
        file_path = "C:\\Users\\yash\\Documents\\LLM_project\\csvchatbot\\output.txt"
        # Save the result to the specified file
        with open(file_path, "w") as file:
            
            file.write(a)
            
            
        def text_to_speech(text):
            
            
            tts = gTTS(text=text, lang='en')
            audio_file = BytesIO()
            tts.write_to_fp(audio_file)
            audio_file.seek(0)
            return audio_file
        

        
        
        with open("output.txt","r") as source_file:
            
            
            content=source_file.read()
        audio_file = text_to_speech(content)
        st.audio(audio_file, format='audio/mp3')
            
        

#       