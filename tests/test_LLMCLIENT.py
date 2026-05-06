"""This file test the LLMCLIENT part."""

import os
from LLMCLIENT.llm_client import generate_response

def test_generate_response():
    """testing generate_response"""
    key = os.getenv("OPENAI_API_KEY")
    api_key = os.getenv("OPENAI_API_KEY", str(key))

    ans = generate_response(openai_key=api_key,
                       user_message="Who is Einstein",
                        context="I am a scientist interested in general relativity",
                        conversation_history=[])

    assert isinstance(ans, str)
    assert len(ans) > 0
