"""SQLite persistence using SQLModel."""
import json
import os
import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select, func

# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------

# DATABASE_URL tells SQLModel where to find (or create) the database file.
# "sqlite:///./doc_pipeline.db" means:
#   - sqlite    → use SQLite (a plain file, no server needed)
#   - ///       → relative path
#   - ./doc_pipeline.db → file called doc_pipeline.db in the current directory
#
# You can override this by setting DATABASE_URL in your .env file.
# For example, to switch to PostgreSQL later you'd just change this to:
#   postgresql://user:password@localhost/doc_pipeline
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./doc_pipeline.db")

# The engine is the low-level connection to the database.
# Think of it as the "phone line" between your Python code and the DB file.
# Everything goes through this object — you create it once and reuse it.
engine = create_engine(DATABASE_URL)


# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------

# SQLModel lets you define a database table as a Python class.
# Each attribute becomes a column in the table.
# Setting table=True tells SQLModel "this is a real DB table, not just a schema".
class Extraction(SQLModel, table=True):
    # Primary key — SQLite auto-increments this for every new row.
    # Optional[int] with default=None means "let the DB assign the value".
    id: Optional[int] = Field(default=None, primary_key=True)

    # The original filename of the uploaded document (e.g. "invoice.pdf").
    filename: str

    # Which extraction schema was used: "default", "invoice", or "contract".
    # Named schema_name instead of schema because schema is a reserved word
    # in SQLAlchemy (the library SQLModel is built on top of).
    schema_name: str

    # The LLM's extracted data stored as a JSON string.
    # We can't store a raw dict in SQLite, so we serialise it to text with
    # json.dumps() on the way in and json.loads() on the way out.
    extracted_data: str

    # Timestamp of when the record was created.
    # default_factory means "call this function to get the default value"
    # — in this case, the current time as an ISO 8601 string.
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Job table
# ---------------------------------------------------------------------------

# A Job tracks the lifecycle of one upload request.
# When a file is uploaded we create a Job immediately and return its id to
# the browser. The actual LLM work happens in the background and updates
# this row as it progresses.
class Job(SQLModel, table=True):
    # UUID string used as the job id.
    # We generate it in Python (not the DB) so we can return it to the client
    # before the row is even inserted. str rather than UUID type keeps it
    # simple — SQLite stores it as plain text.
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)

    # Four possible states, in order:
    #   pending    → job created, worker hasn't picked it up yet
    #   processing → LLM is running
    #   completed  → finished successfully, extraction_id is now set
    #   failed     → something went wrong, error is now set
    status: str = Field(default="pending")

    # The original filename and schema, stored here so the status endpoint
    # can return useful context without needing to join to the Extraction table.
    filename: str
    schema_name: str

    # Set once the job completes — the id of the row in the Extraction table
    # that holds the actual result. Optional because it's None until then.
    extraction_id: Optional[int] = Field(default=None)

    # Set if the job fails. Stores the error message so you can debug it.
    error: Optional[str] = Field(default=None)

    # When the job was first created (i.e. when the file was uploaded).
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def create_db_and_tables() -> None:
    """Create the database file and all tables if they don't exist yet.

    Safe to call every time the app starts — SQLModel only creates tables
    that are missing, it won't overwrite existing data.
    """
    SQLModel.metadata.create_all(engine)


def insert_extraction(filename: str, schema_name: str, extracted_data: dict) -> Extraction:
    """Save one extraction to the database and return it with its assigned id.

    Args:
        filename:       Original name of the uploaded file.
        schema_name:    Which schema was used ("default", "invoice", "contract").
        extracted_data: The structured data dict returned by the LLM.

    Returns:
        The saved Extraction object, now populated with its auto-assigned id.
    """
    record = Extraction(
        filename=filename,
        schema_name=schema_name,
        # Serialise the dict to a JSON string so SQLite can store it.
        extracted_data=json.dumps(extracted_data),
    )

    # Session is a temporary "workspace" for talking to the database.
    # The `with` block makes sure the session is cleanly closed afterwards.
    with Session(engine) as session:
        session.add(record)    # Stage the new row (not written yet)
        session.commit()       # Actually write it to the database file
        session.refresh(record)  # Reload the record so `id` is populated
    return record


def list_extractions(
    limit: int = 10,
    offset: int = 0,
    schema_name: Optional[str] = None,
) -> tuple[list[Extraction], int]:
    """Fetch a page of extractions and the total matching count.

    Args:
        limit:       Max number of rows to return (for pagination).
        offset:      How many rows to skip from the start (for pagination).
                     e.g. offset=10, limit=10 gives you page 2.
        schema_name: If provided, only return extractions with this schema.

    Returns:
        A tuple of (list of Extraction rows, total count of matching rows).
        The total is the full count — not just the current page — so the
        frontend can show "showing 1-10 of 42" style pagination.
    """
    with Session(engine) as session:
        # Build two queries: one for the data, one for the total count.
        # select(Extraction) is like SQL's SELECT * FROM extraction
        query = select(Extraction)
        # func.count() is like SQL's SELECT COUNT(*) FROM extraction
        count_query = select(func.count()).select_from(Extraction)

        # If a schema filter was requested, add a WHERE clause to both queries.
        if schema_name:
            query = query.where(Extraction.schema_name == schema_name)
            count_query = count_query.where(Extraction.schema_name == schema_name)

        # .one() runs the count query and returns the single integer result.
        total = session.exec(count_query).one()

        # .offset() and .limit() add OFFSET / LIMIT to the SQL query.
        # .all() runs it and returns a list of Extraction objects.
        results = session.exec(query.offset(offset).limit(limit)).all()

    return results, total


def get_extraction(extraction_id: int) -> Optional[Extraction]:
    """Fetch a single Extraction row by its id.

    Returns None if not found — same pattern as get_job.
    Used by the /jobs/{id} endpoint to attach result data once a job completes.
    """
    with Session(engine) as session:
        return session.get(Extraction, extraction_id)


def create_job(filename: str, schema_name: str) -> Job:
    """Create a new job row in the 'pending' state and return it.

    Called the moment a file is uploaded, before any LLM work starts.
    The returned job.id is what we send back to the browser immediately.

    Args:
        filename:    Original name of the uploaded file.
        schema_name: Which extraction schema was requested.

    Returns:
        The newly created Job with status="pending".
    """
    job = Job(filename=filename, schema_name=schema_name)
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
    return job


def get_job(job_id: str) -> Optional[Job]:
    """Look up a single job by its id.

    Returns None if no job with that id exists (so the API can return a 404).

    Args:
        job_id: The UUID string returned when the job was created.
    """
    with Session(engine) as session:
        # session.get() is a shortcut for SELECT ... WHERE id = ?
        # It returns None automatically if nothing is found.
        return session.get(Job, job_id)


def update_job_status(
    job_id: str,
    status: str,
    extraction_id: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Update a job's status (and optionally its result or error).

    Called by the background worker as it moves through the pipeline:
      - update_job_status(id, "processing")         ← LLM started
      - update_job_status(id, "completed", extraction_id=42)  ← all done
      - update_job_status(id, "failed", error="...")          ← something broke

    Args:
        job_id:        The UUID of the job to update.
        status:        New status string.
        extraction_id: The id of the completed Extraction row (completed only).
        error:         Error message to store (failed only).
    """
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job is None:
            # Nothing to update — job was somehow deleted. Safe to ignore.
            return
        job.status = status
        if extraction_id is not None:
            job.extraction_id = extraction_id
        if error is not None:
            job.error = error
        session.add(job)   # Re-stage the modified row
        session.commit()   # Write the update to disk
