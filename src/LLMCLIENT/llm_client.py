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
    system_prompt = f"""
                        You are a highly knowledgeable NASA science and space exploration expert.

                        Your responsibilities:
                        - Answer the user's questions clearly, accurately, and concisely.
                        - Use the provided CONTEXT as your primary source of truth.
                        - If the context contains the answer, prioritize it.
                        - If the context is incomplete, you may use general scientific knowledge.
                        - If you are uncertain, explicitly say so instead of hallucinating.
                        - Maintain a professional and educational tone.
                        - Explain technical concepts in simple language when appropriate.
 
                        CONTEXT:
                        {context}
                    """

    logger.info("Building message list")
    messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]

    logger.info("Add conversation histroy (limit to 10)")
    MAX_HISTORY = 10
    if conversation_history:
        messages.extend(conversation_history[-MAX_HISTORY:])

    logger.info("Adding current user prompt")
    messages.append(
                        {
                            "role": "user",
                            "content": user_message
                        }
                    )

    logger.info("Sending request to openapi client")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=300
        )

        assistant_response = response.choices[0].message.content.strip()

        logger.info("Response generated successfully")

        return assistant_response

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return "Sorry, The chatbot encountered an error while generating a response."

if __name__ == "__main__":
    key = os.getenv("OPENAI_API_KEY")
    api_key = os.getenv("OPENAI_API_KEY", str(key))

    ans = generate_response(openai_key=api_key,
                       user_message="Who is Einstein",
                        context="I am a scientist interested in general relativity",
                        conversation_history=[])
    print(ans)
