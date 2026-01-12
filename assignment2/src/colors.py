# coding: utf-8
"""
@File        :   colors.py
@Time        :   2025/10/04 16:22:28
@Author      :   Usercyk
@Description :   Provides ANSI color constants for terminal text formatting
"""
import colorama

# Initialize colorama to automatically reset styles after each print
colorama.init(autoreset=True)

# ANSI escape code constants for colored terminal output
GREEN = colorama.Fore.GREEN    # Green text
RESET = colorama.Fore.RESET    # Reset to default text color
YELLOW = colorama.Fore.YELLOW  # Yellow text
BLUE = colorama.Fore.BLUE       # Blue text
CYAN = colorama.Fore.CYAN      # Cyan text

# Define public exports for the module
__all__ = ["GREEN", "RESET", "YELLOW", "BLUE", "CYAN"]
