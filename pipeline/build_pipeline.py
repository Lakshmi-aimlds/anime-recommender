#This class is our main class in which we build our Vector Store.

from src.data_loader import AnimeDataLoader
from src.vector_store import VectorStoreBuilder
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.custom_exception import CustomException

load_dotenv()

logger = get_logger()

def main():
    try:
        logger.info("Starting the Vector Store build process.")

        # Initialize the data loader
        loader = AnimeDataLoader("data/anime_with_synopsis.csv", "data/anime_processed.csv")
        processed_csv = loader.load_and_process()
        logger.info(f"Data loaded and processed successfully.")

        # Initialize the vector store builder
        vector_builder = VectorStoreBuilder(processed_csv)
        vector_builder.build_and_save_vectorstore()
        logger.info(f"Vector store built and saved successfully.")

        