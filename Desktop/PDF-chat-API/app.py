import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.embeddings import InstructorEmbeddings


def main():
    load_dotenv()

    st.set_page_config(
        page_title="Ask your PDF",
        page_icon=":page_with_curl:",
        layout="wide",
    )
    st.header("Ask your PDF 💭")
    pdf = st.file_uploader("Upload Your PDF", type=['pdf'])

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()


        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.write(chunks)

        model = InstructorEmbeddings()
        # embeddings = model.encode(chunks)
        knowledge_base = FAISS.from_texts(chunks, model)

        user_question = st.text_input("Ask a question about your PDF: ")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature = 0.2,
                max_token=None,
                timeout=None,
                max_retries=3,
            )
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)


# def get_vertex_ai_embeddings(texts):
#     embeddings = []
#     for text in texts:
#         embedding = call


if __name__ =="__main__":
    main()