from typing import Dict, List
from openai import OpenAI
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    logger.info("Creating client")
    client = OpenAI(api_key=openai_key,
                    base_url="https://openai.vocareum.com/v1")

    logger.info("Initiate prompt message, with the context")
    system_prompt = f"""You are an expert at NASA. Use youre expertise to answer
                     the user's question as accurately as possible, and using 
                     the following context:\n {context}"""

    logger.info("Building message list")
    messages = []
    messages.append({
        "role": "system",
        "content": system_prompt
    })



    logger.info("Add conversation histroy")
    messages += conversation_history

    logger.info("Adding current user prompt")
    messages.append({
        "role": "user",
        "content": user_message
    })

    logger.info("Sending request to openapi client")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=50)

    logger.info("Returning response")
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    key = os.getenv("OPENAI_API_KEY")
    api_key = os.getenv("OPENAI_API_KEY", str(key))

    ans = generate_response(openai_key=api_key,
                       user_message="Who is Einstein",
                        context="I am a scientist interested in general relativity",
                        conversation_history=[])
    print(ans)
