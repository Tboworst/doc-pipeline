"""RAG service - handles Pinecone vector storage and semantic search."""
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Create index if it doesn't exist (text-embedding-3-small produces 1536-dim vectors)
index_name = "doc-pipeline"
existing = [idx.name for idx in pc.list_indexes()]
if index_name not in existing:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait until the index is ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)


def embed_text(text: str) -> list:
    """Convert text to embedding vector."""
    
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    #list of numbers 
    return response.data[0].embedding


def save_to_pinecone(extraction_record: dict):
    """Store extraction in Pinecone."""
    try:
        data_str = str(extraction_record["extracted_data"])
        vector = embed_text(data_str)

        metadata = {
            "filename": extraction_record["filename"],
            "schema": extraction_record["schema"],
            "id": extraction_record["id"],
        }

        index.upsert(
            vectors=[
                {
                    "id": str(extraction_record["id"]),
                    "values": vector,
                    "metadata": metadata
                }
            ]
        )
    except Exception as e:
        raise Exception(f"Failed to save to Pinecone: {str(e)}")



def query_pinecone(query_text: str, top_k: int = 5):
    """Search Pinecone for similar extractions."""
    try:
        query_vector = embed_text(query_text)

        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        matches = []
        for match in results["matches"]:
            matches.append({
                "id": match["id"],
                "score": match["score"],
                "metadata": match["metadata"]
            })

        return {
            "success": True,
            "query": query_text,
            "results_count": len(matches),
            "results": matches,
        }
    except Exception as e:
        raise Exception(f"Failed to query Pinecone: {str(e)}")