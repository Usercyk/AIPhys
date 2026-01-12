# coding: utf-8
"""
@File        :   mcp.py
@Time        :   2026/01/09 15:08:41
@Author      :   Usercyk
@Description :   Offer MCP functions
"""
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from src.config import DATA_PATH, LOG_PATH, SRC_PATH


def convert_pdf_to_markdown(file_path_list: List[str]) -> Dict[str, Any]:
    """
    Using multiprocess.
    Convert files to markdown format, for LLM to read the content inside.

    Args:
        file_path_list (str):
            The list of file need to be converted.
    """
    struct_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    log_path = Path(f"{LOG_PATH}/minerU_{struct_time}.log")
    return_code = 1

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(["python",
                                 SRC_PATH+"/convert_worker.py",
                                *file_path_list, DATA_PATH],
                                stdout=log_file,
                                stderr=log_file,
                                bufsize=1,
                                text=True)
        return_code = proc.wait()

    if return_code != 0:
        raise RuntimeError(f"convert_worker failed, see {log_path}")

    return {"description": "All files have been converted into markdown. " +
            "The markdown file have been saved in the store_dir " +
            "with the same file name and extension name '.md'.",
            "store_dir": DATA_PATH}


def read_pure_text_file(file_path: str) -> Dict[str, str]:
    """
    Read the file.

    Args:
        file_path (str): _description_

    Returns:
        Dict[str,Any]: _description_
    """
    path = Path(file_path)
    assert path.suffix in [".txt", ".md", ""]

    content = path.read_text(encoding="utf-8")

    return {"file_content": content}


def create_new_file(file_content: str, file_path: str) -> Dict[str, str]:
    """
    _summary_

    Args:
        file_content (str): _description_
        file_path (str): _description_

    Returns:
        Dict[str,Any]: _description_
    """
    path = Path(file_path)
    assert path.suffix in [".txt", ".md", ""]
    assert not path.exists()

    path.write_text(file_content, encoding="utf-8")

    return {"description": "The content has successfully saved to file",
            "saved_file_path": str(path.resolve())}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "convert_pdf_to_markdown",
            "description": "Convert one or more PDF files into Markdown format " +
            "for LLM consumption. This conversion only change the file's format, " +
            "the content keeps same.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path_list": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "A list of file paths to PDF files that " +
                        "need to be converted into Markdown."
                    }
                },
                "required": ["file_path_list"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_pure_text_file",
            "description": "Read the content of a plain text file and return it as " +
            "a string. Supported file types include .txt, .md, or files without an " +
            "extension.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to a plain text file (.txt, .md, or no " +
                        "extension) to be read using UTF-8 encoding."
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_new_file",
            "description": "Create a new plain text file and write the provided content " +
            "to it using UTF-8 encoding. The operation fails if the target file already " +
            "exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_content": {
                        "type": "string",
                        "description": "The full text content to be written into the new file."
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path where the new file will be created. Supported " +
                        "file types include .txt, .md, or files without an extension. The " +
                        "file must not already exist."
                    }
                },
                "required": ["file_content", "file_path"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

]

FUNCTION_MAP = {"convert_pdf_to_markdown": convert_pdf_to_markdown,
                "read_pure_text_file": read_pure_text_file,
                "create_new_file": create_new_file}
