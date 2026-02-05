#!/usr/bin/env python
"""Check available Google Generative AI models."""

import os
import sys


def check_models():
    """List available Google Generative AI models."""
    try:
        from langchain_google_genai import GoogleGenerativeAI
        from google.generativeai import list_models
    except ImportError:
        print("Error: langchain-google-genai not installed")
        print("Install with: pip install langchain-google-genai")
        return

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        sys.exit(1)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        print("Available Google Generative AI models:")
        print("-" * 60)
        
        for model in list_models():
            print(f"Model: {model.name}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Description: {model.description}")
            print(f"  Input token limit: {model.input_token_limit}")
            print(f"  Output token limit: {model.output_token_limit}")
            print()
            
    except Exception as e:
        print(f"Error listing models: {e}")
        print("\nTroubleshooting:")
        print("- Verify GOOGLE_API_KEY is correct")
        print("- Check API is enabled in Google Cloud Console")
        print("- Ensure you have proper permissions")
        sys.exit(1)


if __name__ == "__main__":
    check_models()
