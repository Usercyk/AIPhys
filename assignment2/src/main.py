# coding: utf-8
"""
@File        :   main.py
@Time        :   2025/10/03 16:28:15
@Author      :   Usercyk
@Description :   Main entry point for the RAG agent.
"""
import os
from typing import Tuple, override

from dotenv import load_dotenv
from colors import BLUE, RESET
from collection import CollectionManager
from llm_client import LLMClientBase, LLMConfig, PreProcessResult


class RAGAgent(LLMClientBase):
    """
    Retrieval-Augmented Generation agent that extends LLMClientBase

    Combines language model capabilities with vector database retrieval
    to provide contextually relevant responses
    """
    WELCOME_MESSAGE = "欢迎使用 RAG 多轮对话客户端！"

    @classmethod
    def create_rag(cls,
                   api_key: str,
                   base_url: str,
                   model: str,
                   collection_manager: CollectionManager) -> 'RAGAgent':
        """
        Creates a configured RAG agent instance

        Args:
            api_key (str): API key for authentication
            base_url (str): Base URL for the API endpoint
            model (str): Model name to use
            collection_manager (CollectionManager): The vector database manager

        Returns:
            RAGAgent: Configured RAG agent instance
        """

        config = cls.ask_user_for_llm_config()

        search_num = 10
        user_input = input("RAG搜索数量 "+BLUE+"[10]"+RESET+":")
        try:
            if user_input.isdigit() and int(user_input) > 0:
                search_num = int(user_input)
            elif user_input != "":
                print("无效输入，RAG搜索数量应该为正整数，将使用默认值10")
        except ValueError:
            print("无效输入，RAG搜索数量应该为正整数，将使用默认值10")
        except EOFError:
            pass

        return cls(api_key, base_url, model, config, search_num, collection_manager)

    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model: str,
                 config: LLMConfig,
                 search_num: int,
                 collection_manager: CollectionManager) -> None:
        """
        Initializes the RAG agent

        Args:
            api_key (str): API key for authentication
            base_url (str): Base URL for the API endpoint
            model (str): Model name to use
            config (LLMConfig): Language model configuration
            collection_manager (CollectionManager): Vector database manager
        """
        super().__init__(api_key, base_url, model, config)
        self.collection_manager = collection_manager
        self.search_num = search_num

    @override
    def preprocess_user_input(self, user_input: str) -> Tuple[str, PreProcessResult]:
        """
        Preprocesses user input by augmenting it with relevant context

        Args:
            user_input (str): Raw input from user

        Returns:
            Tuple[str, PreProcessResult]: Augmented prompt and validation result
        """
        us_input, res = super().preprocess_user_input(user_input)
        if res != PreProcessResult.VALID:
            return us_input, res
        rag_result = self.collection_manager.search(us_input, self.search_num)
        rag_context = self.collection_manager.results_to_str(rag_result)

        prompt_template = """基于以下参考信息回答问题。如果参考信息不足以回答问题，请根据你的知识进行回答。
        其中参考信息包括搜索到的相关内容，标签，以及相似度。
        注意，如果用户的提问和所有参考信息都无关，就根据你自身的信息回答。

        **======参考信息======**：
        {context}

        **======问题======**：
        {question}

        请根据以上信息给出详细、准确的回答："""

        prompt = prompt_template.format(
            context=rag_context,
            question=us_input
        )
        return prompt, res


if __name__ == "__main__":
    # Load environment variables from a .env file if present
    load_dotenv()
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    EF_API_KEY = os.getenv("OPEN_AI_API_KEY")
    assert API_KEY, EF_API_KEY

    db_manager = CollectionManager(
        api_key=EF_API_KEY,
        api_base="http://162.105.151.181/v1",
        model_name="text-embedding-v4"
    )

    rag_agent = RAGAgent.create_rag(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus-latest",
        collection_manager=db_manager
    )

    rag_agent.run()
