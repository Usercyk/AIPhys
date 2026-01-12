# coding: utf-8
"""
@File        :   collection.py
@Time        :   2025/10/03 16:28:15
@Author      :   Usercyk
@Description :   Collection manager for the RAG agent.
"""
from itertools import batched
import json
import os
from typing import Any, Dict, List, Optional, TypedDict

from chromadb import Collection, PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, DefaultEmbeddingFunction
from chromadb.api.types import Metadata, OneOrMany, Document, maybe_cast_one_to_many
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import wikipedia
from wikipedia.exceptions import PageError


class SearchResult(TypedDict):
    """
    Represents a single search result from the vector database

    Attributes:
        content (str): The text content of the matching document
        metadata (Optional[Metadata]): Associated metadata for the document
        similarity (float): Similarity score (1 = identical, 0 = completely different)
    """

    content: str
    metadata: Optional[Metadata]
    similarity: float


class CollectionManager:
    """
    Manages vector collections for RAG (Retrieval-Augmented Generation) applications

    Handles collection creation, document storage, and similarity search
    using ChromaDB vector database with optional OpenAI embeddings
    """

    ID_PREFIX = "id_"

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 model_name: Optional[str] = None,
                 db_path: Optional[str] = None,
                 db_name: Optional[str] = None) -> None:
        """
        Initializes the collection manager

        Args:
            api_key (Optional[str]): OpenAI API key for embeddings
            api_base (Optional[str]): OpenAI API base URL
            model_name (Optional[str]): OpenAI embedding model name
            db_path (Optional[str]): Path to ChromaDB storage
            db_name (Optional[str]): Name of the collection
        """

        if db_path is None:
            db_path = "/home/stu2400011486/assignments/assignment2/.chroma"
        if db_name is None:
            db_name = "Assignment2"

        self.db_client = PersistentClient(path=db_path, settings=Settings(
            allow_reset=True,
            anonymized_telemetry=False
        ))
        self.embedding_function = DefaultEmbeddingFunction()
        if api_key is not None and model_name is not None and api_base is not None:
            self.embedding_function = OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=api_base,
                model_name=model_name
            )
        self.db_name = db_name
        self.collection = self.get_or_create_collection()
        self.count = self.collection.count()

    def add_documents(self,
                      documents: OneOrMany[Document],
                      metadatas: Optional[OneOrMany[Metadata]] = None) -> None:
        """
        Adds documents to the collection

        Args:
            documents (OneOrMany[Document]): Document(s) to add
            metadatas (Optional[OneOrMany[Metadata]]): Optional metadata for documents
        """
        if metadatas is None:
            for batch_doc in tqdm(batched(documents, 10), desc="Adding docs", unit="batch"):
                self._add_documents(list(batch_doc), None)
        else:
            unpack_meta = maybe_cast_one_to_many(metadatas)
            assert unpack_meta is not None
            for batch_doc, batch_meta in tqdm(zip(batched(documents, 10),
                                                  batched(unpack_meta, 10)),
                                              desc="Adding docs",
                                              unit="batch"):
                self._add_documents(list(batch_doc), list(batch_meta))

    def _add_documents(self,
                       documents: OneOrMany[Document],
                       metadatas: Optional[OneOrMany[Metadata]] = None) -> None:
        """
        Adds documents to the collection

        Args:
            documents (OneOrMany[Document]): Document(s) to add
            metadatas (Optional[OneOrMany[Metadata]]): Optional metadata for documents
        """

        unpack_docs = maybe_cast_one_to_many(documents)

        assert unpack_docs is not None

        self.collection.add(
            documents=unpack_docs,
            metadatas=metadatas,
            ids=[
                f"{self.ID_PREFIX}{i + self.count}" for i in range(len(unpack_docs))
            ]
        )
        self.count = self.collection.count()

    def search(self,
               query: str,
               n_results: int = 3) -> List[SearchResult]:
        """
        Searches the collection for similar documents

        Args:
            query (str): Search query text
            n_results (int, optional): Number of results to return. Defaults to 3.

        Returns:
            List[SearchResult]: Ordered list of search results
        """

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        assert results["documents"] is not None
        assert results["metadatas"] is not None
        assert results["distances"] is not None

        return [{
            "content": doc,
            "metadata": meta,
            "similarity": 1 - dist
        } for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )]

    def get_or_create_collection(self) -> Collection:
        """
        Retrieves existing collection or creates a new one

        Returns:
            Collection: ChromaDB collection instance
        """

        # pyright: ignore[reportArgumentType]
        # Because chromadb's type hints are incorrect.
        # I have to ignore the error.
        return self.db_client.get_or_create_collection(
            name=self.db_name,
            embedding_function=self.embedding_function  # type: ignore
        )

    def reset(self) -> None:
        """
        Resets the collection (deletes all documents)
        """
        self.db_client.reset()
        self.collection = self.get_or_create_collection()
        self.count = 0

    @staticmethod
    def results_to_str(results: List[SearchResult]) -> str:
        """
        Formats search results as a readable string

        Args:
            results (List[SearchResult]): Search results to format

        Returns:
            str: Formatted results string
        """
        return "\n".join([
            f"Content: {res['content']}\n"
            f"Metadata: {res['metadata']}\n"
            f"Similarity: {res['similarity']:.4f}\n"
            "-----"
            for res in results
        ])

    def add_json(self, path: str) -> None:
        """
        Add documents in json file.

        Args:
            path (str): The json file path
        """
        with open(path, "r", encoding="utf-8") as fp:
            data: Dict[str, List[Any]] = json.load(fp)
        assert data
        documents = data["documents"]
        metadatas = data["metadatas"]
        # Some api will restrict the batch size
        # So we have to add only one data once a time
        self.add_documents(documents, metadatas)

    def add_text(self, text: str) -> None:
        """
        Add text.

        Args:
            text (str): The content.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=500,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],
        )
        docs = text_splitter.split_text(text)
        self.add_documents(docs)

    def add_text_file(self, path: str) -> None:
        """
        Add text from a file.

        Args:
            path (str): The path to text file.
        """
        with open(path, "r", encoding="utf-8") as fp:
            data = fp.read()
        assert data
        self.add_text(data)

    def add_wikipedia(self, page: str) -> None:
        """
        Add content search from wikipedia.

        Ambiguition: Add the suggests one.
        Not match any page: Add nothing.

        Args:
            page (str): The page title.
        """
        os.environ["http_proxy"] = "http://127.0.0.1:7890"
        os.environ["https_proxy"] = "http://127.0.0.1:7890"

        wikipedia.set_lang("zh")
        print(f"Collecting page {page} from wikipedia")
        try:
            content = wikipedia.page(page, auto_suggest=True).content
            self.add_text(content)
        except PageError:
            pass


if __name__ == "__main__":
    # test
    load_dotenv()
    EF_API_KEY = os.getenv("OPEN_AI_API_KEY")
    db_manager = CollectionManager(
        api_key=EF_API_KEY,
        api_base="http://162.105.151.181/v1",
        model_name="text-embedding-v4"
    )
    # db_manager.add_json(
    #     "/home/stu2400011486/assignments/assignment2/data/rag_database.json")
    # db_manager.add_text_file(
    #     "/home/stu2400011486/assignments/assignment2/data/physics_knowledge.txt"
    # )
    # db_manager.add_wikipedia("Python")
    print(db_manager.search("Python"))
