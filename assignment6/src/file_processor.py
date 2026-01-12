# coding: utf-8
"""
@File        :   file_processor.py
@Time        :   2026/01/09 10:24:17
@Author      :   Usercyk
@Description :   Scan the file and convert it into markdown.
"""
import os
from pathlib import Path

from config import DATA_PATH, TEMP_PATH

from mineru.cli.common import read_fn, prepare_env
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.engine_utils import get_vlm_engine
from mineru.backend.hybrid.hybrid_analyze import doc_analyze as hybrid_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make
from mineru.utils.enum_class import MakeMode


def convert_file_list_to_markdown(file_path_list: str, store_dir: str = DATA_PATH) -> None:
    """
    Convert files to markdown format, for LLM to read the content inside.

    Args:
        file_path_list (str):
            The list of file need to be converted.

        store_dir (str, optional):
            The store location of the converted file. Defaults to DATA_PATH.
    """

    os.environ['MINERU_MODEL_SOURCE'] = "modelscope"

    pdf_bytes_list = [read_fn(path) for path in file_path_list]
    pdf_name_list = [str(Path(path).stem) for path in file_path_list]

    backend = get_vlm_engine(inference_engine='auto', is_async=False)
    md_writer = FileBasedDataWriter(store_dir)

    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        pdf_name = pdf_name_list[idx]

        image_dir, _ = prepare_env(
            output_dir=TEMP_PATH,
            pdf_file_name=pdf_name,
            parse_method="hybrid_auto"
        )

        middle_json, _, _ = hybrid_doc_analyze(
            pdf_bytes,
            image_writer=None,
            backend=backend,
            parse_method="hybrid_auto",
            inline_formula_enable=True,
            server_url=None,
        )

        pdf_info = middle_json["pdf_info"]
        image_dir_name = Path(image_dir).name
        md_content = union_make(pdf_info, MakeMode.MM_MD, image_dir_name)

        md_writer.write_string(f"{pdf_name}.md", md_content)
