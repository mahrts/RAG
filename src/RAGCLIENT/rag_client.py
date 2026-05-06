import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(__file__).parent.parent.parent

    candidate_dirs = [d for d in current_dir.glob("**/*")\
                   if d.is_dir() and ("chroma" in d.name.lower()\
                   or "db" in d.name.lower())]

    for directory in candidate_dirs:
        try:
            client = chromadb.PersistentClient(
                     path=str(directory),
                     settings=Settings(
                                       anonymized_telemetry=False,
                                       allow_reset=True)
                        )

            collections = client.list_collections()

            for collection in collections:

                try:
                    key = f"{directory.name}_{collection.name}"
                    try:
                        count = collection.count()
                    except Exception:
                        count = "unknown"

                    backends[key] = {
                        "directory": str(directory),
                        "collection_name": collection.name,
                        "display_name": f"{directory.name} / {collection.name} ({count} docs)",
                        "count": count}
                    
                except Exception as inner_error:
                    # Handle per-collection failure
                    key = f"{directory.name}_error_{collection.name}"
                    backends[key] = {
                        "directory": str(directory),
                        "collection_name": collection.name,
                        "display_name": f"{directory.name} / {collection.name} (error: {str(inner_error)[:50]})",
                        "count": "error"}
    
        except Exception as e:
            # Handle directory-level failure
            key = f"{directory.name}_unavailable"
            backends[key] = {
                "directory": str(directory),
                "collection_name": None,
                "display_name": f"{directory.name} (error: {str(e)[:50]})",
                "count": "unavailable"
            }
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)
    
    Args:
        chroma_dir (str): The local directory path where the database files are 
            stored. If the directory doesn't exist, Chroma will create it.
            
        collection_name (str): The specific grouping within the database
            to access.

    Returns:
        chromadb.Collection: An object used for querying and managing documents.
    """
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False,
                          allow_reset=True)
    )

    collection = client.get_collection(name=collection_name)
    return collection


def retrieve_documents(collection, query: str, n_results: int = 3,
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering
    
    Args:
        collection (chromadb.Collection): The active ChromaDB collection object 
                                         to search within.
        query (str): The natural language string to search for. This will be 
                     automatically converted into a vector embedding for comparison.
        n_results (int, optional): The maximum number of relevant documents to 
                          return. Defaults to 3.
        mission_filter (str, optional): A metadata filter key. If provided, 
               the search is restricted to documents where the 'mission' metadata 
               field matches this value. Use "all", "*", or "none" to skip filtering.

    Returns:
        Optional[Dict]: A dictionary containing the search results with the 
            following structure:
            {
                'ids': [[str, ...]],
                'documents': [[str, ...]],
                'metadatas': [[dict, ...]],
                'distances': [[float, ...]]
            }
            Returns None if an error occurs during retrieval.
    """
    where_filter = None

    if mission_filter and mission_filter.lower() not in ["all", "*", "none"]:
        where_filter = {
            "mission": mission_filter
        }

    try:
        # Execute query
        results = collection.query(
            query_texts=[query],   # Chroma expects a list of queries
            n_results=n_results,
            where=where_filter     # None = no filter
        )

        return results

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return None

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context
    
    Args:
        documents: A list of strings representing the retrieved text chunks.
        metadatas: A list of dictionaries containing metadata for each chunk 
            (e.g., 'mission', 'source', 'category').

    Returns:
        A single formatted string containing all retrieved context chunks with 
        headers and source attribution. Returns an empty string if no 
        documents are provided.
    """
    if not documents:
        return ""

    context_parts = ["### Retrieved Context:\n"]

    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        meta = meta or {}

        mission = meta.get("mission", "unknown mission")
        source = meta.get("source", "unknown source")
        category = meta.get("category", "general")

        mission_clean = mission.replace("_", " ").title()
        category_clean = category.replace("_", " ").title()

        header = f"[Source {i}] Mission: {mission_clean} | Category: {category_clean} | Source: {source}"
        context_parts.append(header)

        max_chars = 1000
        if len(doc) > max_chars:
            truncated_doc = doc[:max_chars].rstrip() + "..."
        else:
            truncated_doc = doc

        context_parts.append(truncated_doc)
        context_parts.append("")  # spacing between entries

    #return "\n".join(context_parts)
    return "\n".join([s for s in context_parts if isinstance(s, str)])

if __name__ == "__main__":
    D = discover_chroma_backends()
    one_key = list(D.keys())[0]
    print(D[one_key], "\n")

    chroma_dir = D[one_key]["directory"]
    collection_name = D[one_key]["collection_name"]

    collection = initialize_rag_system(chroma_dir=chroma_dir, collection_name=collection_name)
    print("Collection:\n", collection, "\n", type(collection))

    result = retrieve_documents(collection=collection, 
                                query="When was applo 11 launched?")

    print("Result:\n", result, "\n", type(result))

    documents = result["documents"]
    metadatas = result["metadatas"]

    context = format_context(documents=documents, metadatas=metadatas)
    print("Context:\n", type(context), context)
