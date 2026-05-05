"""Storage service for saving extracted document outputs."""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict


class StorageService:
    """Save processed extraction results to disk."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir or os.getenv('OUTPUT_DIR', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: Any, metadata: Dict[str, Any]) -> str:
        """Save extraction result as JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = metadata.get('filename', 'document')
        safe_name = Path(filename).stem
        output_path = self.output_dir / f"{safe_name}_{timestamp}.json"

        payload = {
            'metadata': {
                **metadata,
                'processed_at': datetime.now().isoformat(),
                'model': getattr(result, 'model', None),
                'confidence': getattr(result, 'confidence', None),
            },
            'data': getattr(result, 'structured_data', {}),
        }

        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        return str(output_path)

    def list_outputs(self) -> list[Path]:
        """List all saved output files."""
        return sorted(self.output_dir.glob('*.json'))
