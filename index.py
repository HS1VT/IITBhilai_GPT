import os
import config

os.environ['OPENAI_API_KEY'] = config.api_key

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()

def create_index(file_path: str) -> None:

    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()

    with open('output.txt', 'w', encoding="utf-32") as file:
        file.write(text)

    loader = DirectoryLoader(
        './',
        glob='**/*.txt',
        loader_cls=TextLoader
    )

    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1024,
        chunk_overlap=128
    )

    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        openai_api_key=(config.api_key)
    )

    persist_directory = 'db'

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectordb.persist()

create_index('sample3.pdf')