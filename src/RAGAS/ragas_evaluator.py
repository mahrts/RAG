"""This scripts contains the ragas evaluator."""

import os
from pathlib import Path
import pandas as pd
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import chromadb
import json
from typing import Dict, List, Optional
import logging
from LLMCLIENT.llm_client import generate_response

# RAGAS imports
try:
    from ragas import SingleTurnSample, EvaluationDataset
    from ragas.metrics import (
        BleuScore,
        NonLLMContextPrecisionWithReference,
        ResponseRelevancy,
        Faithfulness,
        RougeScore
    )
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

key = str(os.getenv("OPENAI_API_KEY"))

def evaluate_response_quality(question: str,
                              answer: str, 
                              contexts: List[str],
                              ) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    try:
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.0,
                api_key=key
            )
        )

        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=key
            )
        )

        metrics_list = [
            Faithfulness(llm=evaluator_llm),
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            NonLLMContextPrecisionWithReference(),
            BleuScore(),
            RougeScore()
        ]


        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=" ".join(contexts),
            reference_contexts=contexts
        )

        dataset = EvaluationDataset(samples=[sample])

        results = evaluate(
            dataset=dataset,
            metrics=metrics_list
        )

        return results.to_pandas().iloc[0].to_dict()

    except Exception as e:
        logger.error(f"Error evaluating response with RAGAS: {e}", exc_info=True)
        return {"error": f"Evaluation failed: {str(e)}"}

def get_concatenated_context(question: str, number_of_context: int =5):
    """Getting context from database, given user question"""
    db_path = Path(__file__).parent.parent.parent / "chroma_db_openai"
    client = chromadb.PersistentClient(path=db_path)

    collection = client.get_collection(name="nasa_space_missions_text")

    results = collection.query(
        query_texts=[question],
        n_results=number_of_context
    )

    documents = results.get("documents", [[]])[0]
    single_context_string = "\n\n".join(documents)

    return single_context_string

def run_batch_evaluation(openai_key: str, file_path: str = None):
    """Loads a JSON dataset, evaluates each row, and prints a summary."""
    if file_path is None:
        file_path = Path(__file__).parent / "test_questions.json"

    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    all_results = []

    for entry in data:
        q = entry.get("question")
        c = get_concatenated_context(q)
        a = generate_response(openai_key, user_message=q, context=c,
                      conversation_history={})

        if not q or not a or not c:
            logger.warning(f"Skipping malformed entry: {entry}")
            continue

        logger.info("Evaluating each score...")

        result = evaluate_response_quality(q, a, c)

        if "error" not in result:
            result['category'] = entry.get('category')
            all_results.append(result)

    if not all_results:
        print("No successful evaluations to summarize.")
        return

    df = pd.DataFrame(all_results)

    logger.info("\n--- PER-QUESTION SUMMARY ---")
    print(df[['category', 'faithfulness', 'answer_relevancy', 'bleu_score']])

    logger.info("\n--- AGGREGATE METRICS (MEAN) ---")
    print(df.select_dtypes(include=['number']).mean())
