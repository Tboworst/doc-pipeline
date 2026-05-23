import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .ingestion import IngestionService
from .inference import InferenceService
from .validation import ValidationService
from .rag import save_to_pinecone, query_pinecone
from .database import (
    create_db_and_tables,
    insert_extraction,
    get_extraction,
    list_extractions,
    create_job,
    get_job,
    update_job_status,
)


class QueryRequest(BaseModel):
    query: str


STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def run_pipeline(job_id: str, tmp_path: str, schema: str) -> None:
    """Run the full extraction pipeline for one uploaded file.

    This runs in a background thread *after* the HTTP response has already
    been sent, so the browser never waits on it. It updates the job row in
    the DB as it progresses so the frontend can poll /jobs/{job_id}.

    Args:
        job_id:   The UUID of the Job row created in /upload.
        tmp_path: Path to the temp file holding the uploaded bytes.
        schema:   The extraction schema requested ("default", "invoice", etc.).
    """
    try:
        # Tell the DB (and the polling frontend) we've started.
        update_job_status(job_id, "processing")

        # Run the same pipeline steps that used to block the /upload request.
        ingestion = IngestionService()
        document = ingestion.load(tmp_path)

        inference = InferenceService()
        result = inference.extract(document, schema=schema)

        validator = ValidationService()
        validator.validate(result.structured_data, schema=schema)

        # Persist the extraction result.
        record = insert_extraction(
            filename=document.metadata["filename"],
            schema_name=schema,
            extracted_data=result.structured_data,
        )

        # Best-effort Pinecone — a quota error here should not fail the job.
        try:
            save_to_pinecone({
                "id": record.id,
                "filename": record.filename,
                "schema": schema,
                "extracted_data": result.structured_data,
            })
        except Exception:
            pass

        # Job is done. Link it to the Extraction row so /jobs/{id} can
        # return the actual data when the frontend polls.
        update_job_status(job_id, "completed", extraction_id=record.id)

    except Exception as e:
        # Store the error so the frontend can display it.
        update_job_status(job_id, "failed", error=str(e))

    finally:
        # Always delete the temp file, whether we succeeded or failed.
        # This is the only place it gets cleaned up — /upload deliberately
        # does NOT delete it so the background task can still read it.
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    schema: str = "default",
):
    """Accept a file and return a job_id immediately — no waiting for the LLM.

    The extraction runs in the background. The browser should then poll
    GET /jobs/{job_id} to track progress and retrieve the result.
    """
    valid_schemas = ["default", "invoice", "contract"]
    if schema not in valid_schemas:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid schema. Must be one of: {valid_schemas}"
        )

    # Write uploaded bytes to a temp file.
    # delete=False keeps it alive after this `with` block closes —
    # the background worker needs to read it and will delete it when done.
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Create the job row in the DB (status starts as "pending").
    job = create_job(filename=file.filename, schema_name=schema)

    # Hand the heavy work off to the background.
    # FastAPI runs this function after the response below is sent.
    background_tasks.add_task(run_pipeline, job.id, tmp_path, schema)

    # Respond in milliseconds — browser gets this before the LLM even starts.
    return JSONResponse({
        "success": True,
        "job_id": job.id,
        "filename": file.filename,
        "status": "pending",
    })


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Return the current status of a background extraction job.

    The frontend polls this every ~2 seconds after upload.

    Response always includes: job_id, status, filename, schema.
    When completed: also includes extraction_id and data.
    When failed:    also includes error.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    response = {
        "job_id": job.id,
        "status": job.status,
        "filename": job.filename,
        "schema": job.schema_name,
    }

    if job.status == "completed" and job.extraction_id is not None:
        # Attach the actual extracted data so the browser can display it.
        extraction = get_extraction(job.extraction_id)
        if extraction:
            response["extraction_id"] = extraction.id
            response["data"] = json.loads(extraction.extracted_data)

    if job.status == "failed":
        response["error"] = job.error

    return JSONResponse(response)


@app.get("/results")
async def list_results(limit: int = 10, offset: int = 0, schema: str = None):
    try:
        results, total = list_extractions(limit=limit, offset=offset, schema_name=schema)
        serialised = [
            {
                "id": r.id,
                "filename": r.filename,
                "schema": r.schema_name,
                "extracted_data": json.loads(r.extracted_data),
                "processed_at": r.processed_at,
            }
            for r in results
        ]
        return JSONResponse({
            "success": True,
            "total": total,
            "returned": len(serialised),
            "offset": offset,
            "limit": limit,
            "results": serialised,
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
