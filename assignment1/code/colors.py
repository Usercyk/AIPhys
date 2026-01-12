""" Colors """
import colorama

# Colorama
colorama.init(autoreset=True)

GREEN = colorama.Fore.GREEN
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW
BLUE = colorama.Fore.BLUE
CYAN = colorama.Fore.CYAN

__all__ = ["GREEN", "RESET", "YELLOW", "BLUE", "CYAN"]
