"""
Script to install required LangChain packages for the Serper API integration.
"""

import subprocess
import sys

def install_packages():
    """Install the required packages."""
    packages = [
        "langchain-community",  # For GoogleSerperAPIWrapper
        "langchain",            # For base components
        "langchain-openai"      # For OpenAI integration (used in examples)
    ]
    
    print("Installing required packages...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("All packages installed successfully!")

if __name__ == "__main__":
    install_packages()
