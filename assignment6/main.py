# coding: utf-8
"""
@File        :   main.py
@Time        :   2026/01/09 10:12:14
@Author      :   Usercyk
@Description :   Entrypoint of the program
"""
import os

from src.mcp import TOOLS, FUNCTION_MAP
from src.llm_agent import LLMAgent

from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    LLMAgent(API_KEY,
             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
             model="qwen-plus-latest",
             functions=TOOLS,
             function_map=FUNCTION_MAP).run()
