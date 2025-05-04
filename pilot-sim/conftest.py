import sys
import os

# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

print(f"sys.path after conftest.py: {sys.path}") # Optional: for debugging