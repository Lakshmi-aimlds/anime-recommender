from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

class VectorStoreBuilder:
    def __init__(self, csv_path: str, persist_dir:str="chroma_db"):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    #Build and persist the vector store. This function will load the CSV file, 
    # split the text into chunks, and save the vector store to the specified directory.
    def build_and_save_vectorstore(self):
        # Load the CSV file
        loader = CSVLoader(file_path=self.csv_path, 
                           encoding='utf-8',
                           metadata_columns=[]
                           )
        
        data = loader.load()

        # Split the text into manageable chunks
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = splitter.split_socuments(data)

        db = Chroma.from_documents(
            texts=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        db.persist()

    def load_vectorstore(self):
        # Load the vector store from the specified directory
        db = Chroma(
            persist_directory=self.persist_dir,
            embedding=self.embeddings
        )

