# coding: utf-8
"""
@File        :   setup.py
@Time        :   2025/11/19 14:44:29
@Author      :   Usercyk
@Description :   setup
"""
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(["game_solver_compile.py"]))
