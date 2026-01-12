""" Assignment 1 """

import os
import dataclasses
from typing import Any, Dict, List, Literal, Optional, Tuple
from enum import Enum

from openai import APIError, OpenAI, Stream
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion

from dotenv import load_dotenv

from colors import GREEN, RESET, YELLOW, BLUE, CYAN


class PreProcessResult(Enum):
    """
    The result of preprocessing the user input.
    """
    VALID = 0
    PASS_TO_NEXT_ROUND = 1
    EXIT = 2


@dataclasses.dataclass
class LLMConfig:
    """
    Configuration class for LLMClient to reduce number of parameters in __init__.
    """
    enable_thinking: bool = True
    thinking_budget: Optional[int] = None
    enable_search: bool = True
    stream: bool = True


@dataclasses.dataclass
class StreamingState:
    """
    The state of the streaming response.
    """
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    reasoning_content: str = ""
    answer_content: str = ""
    is_answering: bool = False


class LLMClient:
    """
    The client for LLM mutiple rounds.
    """

    @staticmethod
    def ask_user(prompt: str, default: Literal["y", "n"] = "y") -> bool:
        """
        Prompt the user for input and return their response.

        Args:
            prompt (str): The prompt to display to the user.
            default (Literal["y", "n"], optional):
                The default response if the user just presses enter. Defaults to "y".

        Returns:
            bool: True if the user responded with "y" or "yes", False if "n" or "no".
        """
        choices = "y/n"
        if default == "y":
            choices = BLUE+"[y]"+RESET+"/n"
        if default == "n":
            choices = "y/"+BLUE+"[n]"+RESET

        while True:
            try:
                user_input = input(f"{prompt} {choices}: ").strip().lower()
                if user_input == "":
                    user_input = default
                if user_input in ("y", "yes"):
                    return True
                if user_input in ("n", "no"):
                    return False
                print("Invalid input. Please enter 'y'/'yes' or 'n'/'no'.")
            except EOFError:
                return default == "y"

    @classmethod
    def create(cls, api_key: str,
               base_url: str,
               model: str) -> 'LLMClient':
        """
        A class method to create an instance of LLMClient.

        Args:
            api_key (str): The key to access the API.
            base_url (str): The URL of the model to access.
            model (str): The name of the model to use.

        Returns:
            LLMClient: An instance of LLMClient.
        """
        print(CYAN+"欢迎使用 LLM 多轮对话客户端！")
        print("回车以选择默认选项。")
        stream = LLMClient.ask_user("是否启用流式输出？", default="y")
        enable_thinking = LLMClient.ask_user("是否启用深度思考？", default="y")

        thinking_budget: Optional[int] = None
        if enable_thinking:
            restrict_thinking_budget = LLMClient.ask_user(
                "是否限制思考长度？", default="n")
            if restrict_thinking_budget:
                user_input = input("请输入思考长度限制（以 token 计）：")
                try:
                    if user_input.isdigit() and int(user_input) > 0:
                        thinking_budget = int(user_input)
                    else:
                        print("无效输入，思考长度限制应为正整数。将不限制思考长度。")
                except ValueError:
                    print("无效输入，思考长度限制应为正整数。将不限制思考长度。")

        enable_search = LLMClient.ask_user("是否启用联网搜索？", default="n")

        config = LLMConfig(
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            stream=stream,
            enable_search=enable_search
        )

        return cls(api_key, base_url, model, config)

    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model: str,
                 config: LLMConfig) -> None:
        """
        Initialize and configure the LLM client.

        Args:
            api_key (str): The key to access the API.
            base_url (str): The URL of the model to access.
            model (str): The name of the model to use.
            config (LLMConfig): Configuration object for LLM client.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model: str = model

        self.llm_config: LLMConfig = config

        self.messages = []
        self.completion_configs = self.create_completion_configuration()

        self.clear_console()

    def clear_console(self) -> None:
        """
        Clear the console.
        """
        print("\033c", end="")
        print(CYAN+"欢迎使用 LLM 多轮对话客户端！")
        print("输入 'quit' 退出对话，输入 'clear' 清空历史记录。")
        if (self.llm_config.enable_thinking or
                self.llm_config.enable_search) and not self.llm_config.stream:
            print(
                YELLOW+"Warning: 未开启流式输出的情况下启用深度思考，会有较长的等待时间。")

    def preprocess_user_input(self, user_input: str) -> Tuple[str, PreProcessResult]:
        """
        Preprocess the user input before sending it to the model.

        Args:
            user_input (str): The user input to preprocess.

        Returns:
            Tuple[str, PreProcessResult]:
                The preprocessed user input and the result of the preprocessing.
        """
        user_input = user_input.strip()
        if user_input == "":
            return "", PreProcessResult.PASS_TO_NEXT_ROUND
        if user_input.lower() == "quit":
            return "", PreProcessResult.EXIT
        if user_input.lower() == "clear":
            self.clear_console()
            return "", PreProcessResult.PASS_TO_NEXT_ROUND
        return user_input.strip(), PreProcessResult.VALID

    def create_completion_configuration(self) -> Dict[str, Any]:
        """
        Create the configuration for completion request.

        Returns:
            Dict[str, Any]: The configuration for completion request.
        """
        config: Dict[str, Any] = {
            "model": self.model,
        }

        if self.llm_config.stream:
            config["stream"] = True
            config["stream_options"] = {"include_usage": True}

        extra_body: Dict[str, Any] = {}
        if self.llm_config.enable_thinking:
            extra_body["enable_thinking"] = True
            if self.llm_config.thinking_budget is not None:
                extra_body["thinking_budget"] = self.llm_config.thinking_budget
        if self.llm_config.enable_search:
            extra_body["enable_search"] = True
            extra_body["search_options"] = {"search_strategy": "pro"}

        config["extra_body"] = extra_body

        return config

    def process_chunk(self,
                      chunk: ChatCompletionChunk,
                      conversation_idx: int,
                      state: StreamingState) -> StreamingState:
        """
        Process a single chunk of the streaming response.

        Args:
            chunk (ChatCompletionChunk): The chunk to process.
            conversation_idx (int): The index of the current conversation.
            state (StreamingState): The state of the conversation.

        Returns:
            StreamingState: The updated state of the conversation.
        """
        if chunk.choices:
            delta = chunk.choices[0].delta
            reasoning_content: Optional[str] = getattr(
                delta, 'reasoning_content', None)
            if reasoning_content is not None:
                print(reasoning_content, end='', flush=True)
                state.reasoning_content += reasoning_content
            else:
                if delta.content != "" and not state.is_answering:
                    print(GREEN+f"\n\n[{conversation_idx}] Answer: ")
                    state.is_answering = True
                print(delta.content, end='', flush=True)
                state.answer_content += "" if delta.content is None else delta.content
        elif chunk.usage:
            print(GREEN+f"\n\n[{conversation_idx}] System: ")
            print(f"输入 Tokens: {chunk.usage.prompt_tokens}")
            state.total_input_tokens += chunk.usage.prompt_tokens

            print(f"输出 Tokens: {chunk.usage.completion_tokens}")
            state.total_output_tokens += chunk.usage.completion_tokens

            print(f"总计 Tokens: {chunk.usage.total_tokens}")
            state.total_tokens += chunk.usage.total_tokens

        return state

    def run_streaming(self) -> None:
        """
        Run the LLM client in streaming mode.
        """
        # Initialize state variables
        state: StreamingState = StreamingState()

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        conversation_idx: int = 1

        while True:
            user_input = input(GREEN+f"[{conversation_idx}] You: "+RESET)
            user_input, result = self.preprocess_user_input(user_input)
            if result == PreProcessResult.EXIT:
                break
            if result == PreProcessResult.PASS_TO_NEXT_ROUND:
                conversation_idx += 1
                continue

            user_msg: ChatCompletionMessageParam = {
                "role": "user", "content": user_input}
            messages.append(user_msg)

            try:
                completion: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
                    messages=messages,
                    **self.completion_configs
                )
                if self.llm_config.enable_thinking:
                    print(GREEN+f"\n[{conversation_idx}] Think:")

                # Process all chunks
                for chunk in completion:
                    state = self.process_chunk(chunk, conversation_idx, state)

                messages.append(
                    {"role": "assistant", "content": state.answer_content})
                print("\n")
                conversation_idx += 1

                # Reset state for next conversation
                state.reasoning_content = ""
                state.answer_content = ""
                state.is_answering = False

            except APIError as e:
                print(f"API 请求失败: {e}")
                break

        print(GREEN+"\n\n[Total] System: ")
        print(f"输入 Tokens: {state.total_input_tokens}")
        print(f"输出 Tokens: {state.total_output_tokens}")
        print(f"总计 Tokens: {state.total_tokens}")

    def run_not_streaming(self) -> None:
        """
        Run the LLM client in non-streaming mode.
        """
        answer_content: str = ""
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        conversation_idx: int = 1

        while True:

            user_input = input(GREEN+f"[{conversation_idx}] You: "+RESET)
            user_input, result = self.preprocess_user_input(user_input)
            if result == PreProcessResult.EXIT:
                break
            if result == PreProcessResult.PASS_TO_NEXT_ROUND:
                conversation_idx += 1
                continue

            user_msg: ChatCompletionMessageParam = {
                "role": "user", "content": user_input}
            messages.append(user_msg)

            try:
                completion: ChatCompletion = self.client.chat.completions.create(
                    messages=messages,
                    **self.completion_configs
                )

                if self.llm_config.enable_thinking:
                    reasoning_content: Optional[str] = getattr(
                        completion.choices[0].message, 'reasoning_content', None)
                    print(GREEN+f"\n[{conversation_idx}] Think:")
                    print(reasoning_content)

                print(GREEN+f"\n\n[{conversation_idx}] Answer: ")
                answer_content: str = getattr(
                    completion.choices[0].message, 'content', "")
                print(answer_content)

                assert completion.usage is not None
                print(GREEN+f"\n\n[{conversation_idx}] System: ")
                print(f"输入 Tokens: {completion.usage.prompt_tokens}")
                print(f"输出 Tokens: {completion.usage.completion_tokens}")
                print(f"总计 Tokens: {completion.usage.total_tokens}")

                messages.append(
                    {"role": "assistant", "content": answer_content})
                print("\n")
                conversation_idx += 1

            except APIError as e:
                print(f"API 请求失败: {e}")
                break
            except AssertionError:
                print("API 返回的 usage 信息为空，无法统计 Tokens 使用情况。")
                break

    def run(self) -> None:
        """
        Run the LLM client for multiple rounds of interaction.
        """
        if self.llm_config.stream:
            self.run_streaming()
        else:
            self.run_not_streaming()


if __name__ == "__main__":
    # Load environment variables from a .env file if present
    load_dotenv()
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    if API_KEY is None:
        raise ValueError("DASHSCOPE_API_KEY environment variable is not set.")

    llm_client = LLMClient.create(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus-latest")

    llm_client.run()
