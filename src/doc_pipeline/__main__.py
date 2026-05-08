"""Entry point for `python -m doc_pipeline`."""
import sys
from .cli import cli

if __name__ == '__main__':
    # Run FastAPI server with: python -m doc_pipeline server
    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        import uvicorn
        from .api import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        cli()