# coding: utf-8
"""
@File        :   conver_worker.py
@Time        :   2026/01/09 14:28:31
@Author      :   Usercyk
@Description :   The worker file for file_processor
"""
import sys

from file_processor import convert_file_list_to_markdown

file_list = sys.argv[1:-1]
store_dir = sys.argv[-1]

if __name__=="__main__":
    convert_file_list_to_markdown(file_list, store_dir)
