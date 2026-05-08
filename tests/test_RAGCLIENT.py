"""This script tests the RAGCLIENT part"""

import chromadb
from RAGCLIENT.rag_client import discover_chroma_backends
from RAGCLIENT.rag_client import initialize_rag_system
from RAGCLIENT.rag_client import retrieve_documents
from RAGCLIENT.rag_client import format_context


D = discover_chroma_backends()
one_key = list(D.keys())[0]

chroma_dir = D[one_key]["directory"]
collection_name = D[one_key]["collection_name"]
collection, _, _ = initialize_rag_system(chroma_dir=chroma_dir, collection_name=collection_name)

result = retrieve_documents(collection=collection, 
                                query="When was applo 11 launched?")

documents = result["documents"][0] if result["documents"] else []
metadatas = result["metadatas"][0] if result["metadatas"] else []

context = format_context(documents=documents, metadatas=metadatas)

def test_discover_chroma_backends():
    #checking chroma backed keys
    one_key = list(D.keys())[0]
    backends = D[one_key]
    backends_key = list(backends.keys())

    assert "directory" in backends_key
    assert "collection_name" in backends_key
    assert "display_name" in backends_key
    assert "count" in backends_key

def test_initialize_rag_system():
    if collection is not None:
        assert isinstance(collection, chromadb.api.models.Collection.Collection)

def test_retrieve_documents():
    assert isinstance(result, dict)

    result_keys = list(result.keys())
    assert "ids" in result_keys
    assert "embeddings" in result_keys
    assert "documents" in result_keys
    assert "data" in result_keys
    assert "metadatas" in result_keys

def test_format_context():
    assert isinstance(context, str)
