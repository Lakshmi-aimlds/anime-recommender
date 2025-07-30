from langchain.chains import Retrieval_QA
from langchain_groq import ChatGroq
from src.prompt_template import get_anime_prompt

class AnimeRecommender:
    def __init__(self, retriever, api_key:str, model_name:str):
        self.llm = ChatGroq(api_key=api_key,
                            model_name=model_name,
                            temperature=0
                            )
        self.prompt = get_anime_prompt()

        self.qa_chain = Retrieval_QA.from_chain_type(
            llm=self.llm,
            chain_type = "stuff",  #"stuff" retrieves all documents and concatenates them
            retriever = retriever,
            return_source_documents = True,
            chain_type_kwargs = {"prompt": self.prompt}
        )

    def get_recommendation(self, query:str):
        result = self.qa_chain({"query":query})
        return result['result'] 