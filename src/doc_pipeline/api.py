from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from .ingestion import IngestionService
from .inference import InferenceService
from .validation import ValidationService
from .rag import save_to_pinecone, query_pinecone
import os
import tempfile
from pathlib import Path
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str

EXTRACTIONS_STORE = []
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), schema: str = "default"):
    try:
        # Step 1: Validate schema
        valid_schemas = ["default", "invoice", "contract"]
        if schema not in valid_schemas:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid schema. Must be one of: {valid_schemas}"
            )

        # Step 2: Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Step 3: Run extraction pipeline
            ingestion = IngestionService()
            document = ingestion.load(tmp_path)

            inference = InferenceService()
            result = inference.extract(document, schema=schema)

            validator = ValidationService()
            validator.validate(result.structured_data, schema=schema)

            # Step 4: Store the result in memory and Pinecone
            extraction_record = {
                "id": len(EXTRACTIONS_STORE),
                "filename": document.metadata["filename"],
                "schema": schema,
                "extracted_data": result.structured_data,
            }
            EXTRACTIONS_STORE.append(extraction_record)
            save_to_pinecone(extraction_record)

            # Step 5: Return the data
            return JSONResponse({
                "success": True,
                "extraction_id": extraction_record["id"],
                "filename": extraction_record["filename"],
                "data": result.structured_data,
            })
        finally:
            # Cleanup temp file
            os.unlink(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/results")
async def list_results(limit: int = 10, offset: int = 0, schema: str = None):
    #all in try block because it could raise an error and we wouldn't be able to handle without it 
    try:
        # Start with all extractions
        results = EXTRACTIONS_STORE

        # Filter by schema if provided
        if schema:
            results = [r for r in results if r["schema"] == schema]

        # Paginate the results
        paginated = results[offset:offset + limit]

        # Return formatted response
        return JSONResponse({
            "success": True,
            "total": len(results),
            "returned": len(paginated),
            "offset": offset,
            "limit": limit,
            "results": paginated,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def search_results(request: QueryRequest):
    try:
        result = query_pinecone(request.query)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))