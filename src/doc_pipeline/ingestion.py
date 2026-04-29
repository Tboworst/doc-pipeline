"""Document ingestion service - handles loading, format detection, and chunking."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Constants
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB - chunk if larger
CHUNK_SIZE = 500 * 1024          # 500KB chunks
SUPPORTED_TYPES = {'.txt', '.md', '.json', '.csv', '.pdf'}


@dataclass
class Document:
    """Represents a loaded document."""
    content: str
    file_path: str
    file_type: str
    metadata: dict
    chunks: Optional[list] = None  # None if not chunked, list if chunked


class IngestionService:
    """Service for loading and processing documents."""
    def load(self, file_path: str) -> Document:
        """Load a document, chunking if necessary."""
        #create a path variable that maps to the file path
        path = Path(file_path)
        # 1. Validate file exist
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        # 2. Detect file type
        file_type = path.suffix.lower()
        if file_type not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported file type: {file_type}")
        # 3. Read content based on type
        content = self._read_file([path,file_type])
        chunks = None
        #to check the size we want to use len and check the content
        if len(content.encode('utf-8')) > MAX_FILE_SIZE:
            chunks = self._chunk_if_needed(content)
        
        return Document(
        content=content,
        file_path=str(path.absolute()),
        file_type=file_type,
        metadata={
            'filename': path.name,
            'size_bytes': path.stat().st_size,
        },
        chunks=chunks
    )

    
    def _detect_format(self, path: Path) -> str:
        """Detect file format from extension."""
        file_type = path.suffix.lower()
        if file_type not in SUPPORTED_TYPES:
            raise ValueError(
            f"Unsupported file type: {file_type}. "
            f"Supported types: {', '.join(SUPPORTED_TYPES)}"
        )
    
        return file_type

    
    def _read_file(self, path: Path, file_type: str) -> str:
        if file_type in {'.txt', '.md'}:
        # Plain text - read directly
            return path.read_text(encoding='utf-8')
    
        elif file_type == '.json':
        # JSON - read as text (LLM will parse)
            return path.read_text(encoding='utf-8')
    
        elif file_type == '.csv':
        # CSV - read as text
            return path.read_text(encoding='utf-8')

        #Pdf as its own method since its not a text 
        elif file_type == '.pdf':
        
            return self._read_pdf(path)
    
        else:
            raise ValueError(f"Cannot read file type: {file_type}")
    
    def _chunk_if_needed(self, content: str) -> list:
        """Split content into chunks if too large.
        
        Args:
            content: The full document content
            
        Returns:
            List of content chunks
        """
        content_bytes = content.encode('utf-8')
        chunks = []
        
        # Split into chunks of CHUNK_SIZE
        for i in range(0, len(content_bytes), CHUNK_SIZE):
            chunk = content_bytes[i:i + CHUNK_SIZE].decode('utf-8', errors='ignore')
            chunks.append(chunk)
        
        return chunks
    
    def _read_pdf(self, path: Path) -> str:
        """Extract text from PDF using PyPDF2.
        
        Args:
            path: Path object to the PDF file
            
        Returns:
            Extracted text from all pages
        """
        from PyPDF2 import PdfReader
        
        reader = PdfReader(path)
        text_parts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:  # Skip None or empty pages
                text_parts.append(text)
        
        return '\n\n'.join(text_parts)