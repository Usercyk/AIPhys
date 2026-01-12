# coding: utf-8
"""
@File        :   llm_agent.py
@Time        :   2026/01/09 10:29:20
@Author      :   Usercyk
@Description :   The LLM agent
"""
import json
from enum import Enum
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass

from src.colors import CYAN, GREEN, RESET, YELLOW

from openai import APIError, OpenAI, Stream
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionChunk


class PreProcessResult(Enum):
    """
    Represents the result of preprocessing user input
    """
    VALID = 0
    CONTINUE = 1
    EXIT = 2


@dataclass
class StreamingState:
    """
    Tracks state during streaming of LLM responses
    """
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    reasoning_content: str = ""
    answer_content: str = ""
    is_answering: bool = False
    function_name: str = ""
    function_args: str = ""
    is_function_calling: bool = False
    function_call_id: str = ""
    finish_reason: Optional[str] = None


class LLMAgent:
    """
    The agent to call the the llm online
    """
    WELCOME_MESSAGE = "欢迎使用AI Reading Helper"

    def clear_console(self) -> None:
        """
        Clears the console and displays welcome message
        """
        print("\033c", end="")
        print(CYAN+self.WELCOME_MESSAGE)
        print("输入 'quit' 退出对话，输入 'clear' 清空历史记录。")

    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model: str,
                 functions: Optional[List[Dict[str, Any]]] = None,
                 function_map: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the LLM client

        Args:
            api_key (str): API key for authentication
            base_url (str): Base URL for the API endpoint
            model (str): Model name to use
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model: str = model
        self.functions = functions
        self.function_map = function_map

        self.messages = []
        self.completion_config = self.create_completion_configuration()

        self.clear_console()

    def create_completion_configuration(self) -> Dict[str, Any]:
        """
        Creates configuration for LLM completion requests

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        config: Dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {"enable_thinking": True}
        }
        if self.functions is not None:
            config["tools"] = self.functions

        return config

    def preprocess_user_input(self, user_input: str) -> Tuple[str, PreProcessResult]:
        """
        Preprocesses and validates user input

        Args:
            user_input (str): Raw input from user

        Returns:
            Tuple[str, PreProcessResult]: Processed input and validation result
        """
        user_input = user_input.strip()
        if user_input == "":
            return "", PreProcessResult.CONTINUE
        if user_input.lower() == "quit":
            return "", PreProcessResult.EXIT
        if user_input.lower() == "clear":
            self.clear_console()
            return "", PreProcessResult.CONTINUE
        return user_input.strip(), PreProcessResult.VALID

    def process_chunk(self,
                      chunk: ChatCompletionChunk,
                      conversation_idx: int,
                      state: StreamingState) -> StreamingState:
        """
        Processes a streaming response chunk

        Args:
            chunk (ChatCompletionChunk): Response chunk from API
            conversation_idx (int): Current conversation index
            state (StreamingState): Current streaming state

        Returns:
            StreamingState: Updated streaming state
        """
        if chunk.choices:
            delta = chunk.choices[0].delta
            reasoning_content: Optional[str] = getattr(
                delta, 'reasoning_content', None)
            if reasoning_content is not None:
                print(reasoning_content, end='', flush=True)
                state.reasoning_content += reasoning_content
            else:
                function_call_delta = getattr(delta, "tool_calls", None)
                if function_call_delta is not None:
                    if not state.is_function_calling:
                        state.is_function_calling = True
                        print(
                            YELLOW+f"\n\n[{conversation_idx}] Function calling: ")
                    name = function_call_delta[0].function.name
                    name = "" if name is None else name
                    args = function_call_delta[0].function.arguments
                    args = "" if args is None else args

                    print(name, args, sep='', end='', flush=True)

                    state.function_name += name
                    state.function_args += args
                    state.function_call_id = function_call_delta[0].id

                elif delta.content is not None:
                    if delta.content != "" and not state.is_answering:
                        print(GREEN+f"\n\n[{conversation_idx}] Answer: ")
                        state.is_answering = True
                    print(delta.content, end='', flush=True)
                    state.answer_content += "" if delta.content is None else delta.content

            state.finish_reason = chunk.choices[0].finish_reason

        elif chunk.usage:
            print(GREEN+f"\n\n[{conversation_idx}] System: ")
            print(f"输入 Tokens: {chunk.usage.prompt_tokens}")
            state.total_input_tokens += chunk.usage.prompt_tokens

            print(f"输出 Tokens: {chunk.usage.completion_tokens}")
            state.total_output_tokens += chunk.usage.completion_tokens

            print(f"总计 Tokens: {chunk.usage.total_tokens}")
            state.total_tokens += chunk.usage.total_tokens

        return state

    def handle_function_calling(self, func_name: str, func_args: str, tool_call_id: str) -> Dict[str, Any]:
        """
        _summary_

        Args:
            func_name (str): _description_
            func_args (str): _description_

        Returns:
            Dict[str, Any]: _description_
        """
        func = self.function_map.get(func_name, None)
        if func is not None:
            try:
                args = json.loads(func_args)
                res = func(**args)
                if res is None:
                    res = {"success": True,
                           "description": "This is a void function, No Return."}
                res_str = json.dumps(res)
                return {
                    "role": "tool",
                    "content": res_str,
                    "tool_call_id": tool_call_id
                }
            except Exception as e:  # pylint: disable=all
                print(e)
                return {
                    "role": "tool",
                    "content": {"error": str(e)},
                    "tool_call_id": tool_call_id
                }
        return {
            "role": "tool",
            "content": {"error": "No function named "+func_name},
            "tool_call_id": tool_call_id
        }

    def run(self) -> None:
        """
        Start the LLM Agent
        """
        state = StreamingState()
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": "你是一个乐于助人的AI助手，没有后续要求时，回复请使用中文"}
        ]
        conversation_idx: int = 1

        need_input = True

        while True:
            if need_input:
                user_input = input(GREEN+f"[{conversation_idx}] You: "+RESET)
                user_input, result = self.preprocess_user_input(user_input)
                if result == PreProcessResult.EXIT:
                    break
                if result == PreProcessResult.CONTINUE:
                    conversation_idx += 1
                    continue

                user_msg: ChatCompletionMessageParam = {
                    "role": "user", "content": user_input}
                messages.append(user_msg)
            else:
                need_input = True

            try:
                completion: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
                    messages=messages,
                    **self.completion_config
                )
                print(GREEN+f"\n[{conversation_idx}] Think:")

                # Process all chunks
                for chunk in completion:
                    state = self.process_chunk(chunk, conversation_idx, state)

                if state.is_function_calling:
                    tool_msg = self.handle_function_calling(
                        state.function_name, state.function_args, state.function_call_id)
                    messages.append({"role": "assistant",
                                     "tool_calls": [{'id': state.function_call_id,
                                                     "type": "function",
                                                     "function": {
                                                         "arguments": state.function_args,
                                                         "name": state.function_name
                                                     }
                                                     }]})
                    messages.append(tool_msg)
                    need_input = False

                    print("\n")

                    state.reasoning_content = ""
                    state.answer_content = ""
                    state.is_answering = False
                    state.function_name = ""
                    state.function_args = ""
                    state.is_function_calling = False

                    continue

                if state.is_answering:
                    messages.append(
                        {"role": "assistant", "content": state.answer_content})

                print("\n")
                conversation_idx += 1

                # Reset state for next conversation
                state.reasoning_content = ""
                state.answer_content = ""
                state.is_answering = False
                state.function_name = ""
                state.function_args = ""
                state.is_function_calling = False

            except APIError as e:
                print(f"API 请求失败: {e}")
                print(messages)
                break

        print(GREEN+"\n\n[Total] System: ")
        print(f"输入 Tokens: {state.total_input_tokens}")
        print(f"输出 Tokens: {state.total_output_tokens}")
        print(f"总计 Tokens: {state.total_tokens}")
