"""LLM inference service - handles OpenAI extraction and chunk merging."""
import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class ExtractionResult:
    """Result returned by the inference service."""
    raw_output: str
    structured_data: dict
    confidence: float
    model: str


class InferenceService:
    """Service for calling OpenAI and extracting structured data."""

    # Define target schemas that the model can extract.
    SCHEMAS = {
        'default': {
            'description': 'Extract a title, summary, entities, and topics from the document.',
            'fields': ['title', 'summary', 'entities', 'topics']
        },
        'invoice': {
            'description': 'Extract invoice fields from the document.',
            'fields': ['invoice_number', 'date', 'vendor', 'total', 'line_items']
        },
        'contract': {
            'description': 'Extract contract terms and parties from the document.',
            'fields': ['parties', 'effective_date', 'term', 'obligations', 'termination']
        }
    }

    def __init__(self, model: str = None):
        # Load API key from environment and make sure it is present.
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('OPENAI_API_KEY must be set in the environment')

        # Create an OpenAI client instance.
        self.client = OpenAI(api_key=api_key)
        # Allow overriding the model with an environment variable or constructor arg.
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4o')

    def extract(self, document: Any, schema: str = 'default') -> ExtractionResult:
        """Extract structured data from the document."""
        # Pick the schema configuration from the supported schemas.
        schema_config = self.SCHEMAS.get(schema, self.SCHEMAS['default'])

        # If the document has been chunked, extract each chunk separately.
        if document.chunks:
            chunk_results = [self._extract_chunk(chunk, schema_config) for chunk in document.chunks]
            return self._merge_chunked_results(chunk_results, schema_config)

        # Otherwise extract from the full document content.
        return self._extract_chunk(document.content, schema_config)

    def _extract_chunk(self, content: str, schema_config: dict) -> ExtractionResult:
        # Build a prompt that tells the model exactly what to extract.
        prompt = self._build_prompt(content, schema_config)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': 'You are a document extraction assistant.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.0,
            max_tokens=1500,
        )

        # The raw model output may contain JSON or extra text.
        raw_output = response.choices[0].message.content
        structured_data = self._parse_json(raw_output)

        return ExtractionResult(
            raw_output=raw_output,
            structured_data=structured_data,
            confidence=0.9,  # Placeholder that can be improved later.
            model=self.model,
        )

    def _build_prompt(self, content: str, schema_config: dict) -> str:
        # Create a simple extraction prompt with instructions and the document text.
        fields = ', '.join(schema_config['fields'])
        return (
            f"Extract the following fields from the document text:\n"
            f"{schema_config['description']}\n\n"
            f"Fields: {fields}\n\n"
            f"Document text:\n" + content + "\n\n"
            "Return the answer as a single valid JSON object with the exact field names. "
            "If a field is not present, return an empty string or empty list."
        )

    def _parse_json(self, raw_text: str) -> dict:
        # Try parsing the response as JSON directly.
        raw_text = raw_text.strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # If parsing fails, try to find the first JSON object in the text.
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw_text[start:end+1])
                except json.JSONDecodeError:
                    pass

        # If we still can't parse it, return the raw text for debugging.
        return {'raw_text': raw_text}

    def _merge_chunked_results(self, results: list, schema_config: dict) -> ExtractionResult:
        # Prepare a merged container for each schema field.
        merged = {
            field: [] if field in ['entities', 'topics', 'line_items', 'parties', 'obligations'] else []
            for field in schema_config['fields']
        }

        # Combine values from each chunk result.
        for result in results:
            for field in schema_config['fields']:
                value = result.structured_data.get(field)
                if value is None:
                    continue

                if isinstance(value, list):
                    merged[field].extend(value)
                elif isinstance(value, str):
                    if value.strip():
                        merged[field].append(value.strip())
                else:
                    merged[field].append(value)

        # Normalize merged values into final output shape.
        normalized = {}
        for field in schema_config['fields']:
            if field in ['entities', 'topics', 'line_items', 'parties', 'obligations']:
                normalized[field] = merged[field]
            else:
                normalized[field] = ' '.join(dict.fromkeys(merged[field])) if merged[field] else ''

        # Keep the raw outputs from each chunk for debugging if needed.
        combined_raw = '\n---\n'.join(result.raw_output for result in results)

        return ExtractionResult(
            raw_output=combined_raw,
            structured_data=normalized,
            confidence=0.9,
            model=self.model,
        )
