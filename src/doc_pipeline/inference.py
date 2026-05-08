"""LLM inference service - handles multiple LLM providers (OpenAI, Claude)."""
import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ExtractionResult:
    """Result returned by the inference service."""
    raw_output: str
    structured_data: dict
    confidence: float
    model: str
    provider: str


class InferenceService:
    """Service for calling LLM APIs and extracting structured data."""

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

    def __init__(self, provider: str = None, model: str = None):
        # Determine which LLM provider to use
        self.provider = provider or os.getenv('LLM_PROVIDER', 'groq')

        # Set up the appropriate client based on provider
        if self.provider == 'openai':
            self._setup_openai()
        elif self.provider == 'claude':
            self._setup_claude()
        elif self.provider == 'groq':
            self._setup_groq()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        # Set model
        self.model = model or os.getenv('LLM_MODEL', self._get_default_model())

    def _setup_openai(self):
        """Set up OpenAI client."""
        try:
            from openai import OpenAI
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError('OPENAI_API_KEY must be set for OpenAI provider')
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

    def _setup_claude(self):
        """Set up Claude client."""
        try:
            from anthropic import Anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError('ANTHROPIC_API_KEY must be set for Claude provider')
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")

    def _setup_groq(self):
        """Set up Groq client."""
        try:
            from groq import Groq
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError('GROQ_API_KEY must be set for Groq provider')
            self.client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError("Groq package not installed. Run: pip install groq")

    def _get_default_model(self) -> str:
        """Get default model for the current provider."""
        if self.provider == 'openai':
            return 'gpt-4o'
        elif self.provider == 'claude':
            return 'claude-3-haiku-20240307'
        elif self.provider == 'groq':
            return 'llama-3.1-8b-instant'
        return 'llama3-8b-8192'

    def extract(self, document: Any, schema: str = 'default') -> ExtractionResult:
        """Extract structured data from the document."""
        schema_config = self.SCHEMAS.get(schema, self.SCHEMAS['default'])

        if document.chunks:
            chunk_results = [self._extract_chunk(chunk, schema_config) for chunk in document.chunks]
            return self._merge_chunked_results(chunk_results, schema_config)

        return self._extract_chunk(document.content, schema_config)

    def _extract_chunk(self, content: str, schema_config: dict) -> ExtractionResult:
        """Extract from a single chunk of content."""
        prompt = self._build_prompt(content, schema_config)

        if self.provider == 'openai':
            return self._extract_openai(prompt)
        elif self.provider == 'claude':
            return self._extract_claude(prompt)
        elif self.provider == 'groq':
            return self._extract_groq(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _extract_openai(self, prompt: str) -> ExtractionResult:
        """Extract using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': 'You are a document extraction assistant.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.0,
            max_tokens=1500,
        )

        raw_output = response.choices[0].message.content
        structured_data = self._parse_json(raw_output)

        return ExtractionResult(
            raw_output=raw_output,
            structured_data=structured_data,
            confidence=0.9,
            model=self.model,
            provider='openai',
        )

    def _extract_claude(self, prompt: str) -> ExtractionResult:
        """Extract using Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            temperature=0.0,
            system="You are a document extraction assistant. Always respond with valid JSON.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        raw_output = response.content[0].text
        structured_data = self._parse_json(raw_output)

        return ExtractionResult(
            raw_output=raw_output,
            structured_data=structured_data,
            confidence=0.9,
            model=self.model,
            provider='claude',
        )

    def _extract_groq(self, prompt: str) -> ExtractionResult:
        """Extract using Groq API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': 'You are a document extraction assistant. Always respond with valid JSON.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.0,
            max_tokens=1500,
        )

        raw_output = response.choices[0].message.content
        structured_data = self._parse_json(raw_output)

        return ExtractionResult(
            raw_output=raw_output,
            structured_data=structured_data,
            confidence=0.9,
            model=self.model,
            provider='groq',
        )

    def _build_prompt(self, content: str, schema_config: dict) -> str:
        """Build extraction prompt."""
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
        """Parse JSON from LLM response."""
        raw_text = raw_text.strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw_text[start:end+1])
                except json.JSONDecodeError:
                    pass

        return {'raw_text': raw_text}

    def _merge_chunked_results(self, results: list, schema_config: dict) -> ExtractionResult:
        """Merge results from multiple chunks."""
        merged = {
            field: [] if field in ['entities', 'topics', 'line_items', 'parties', 'obligations'] else []
            for field in schema_config['fields']
        }

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

        normalized = {}
        for field in schema_config['fields']:
            if field in ['entities', 'topics', 'line_items', 'parties', 'obligations']:
                normalized[field] = merged[field]
            else:
                normalized[field] = ' '.join(dict.fromkeys(merged[field])) if merged[field] else ''

        combined_raw = '\n---\n'.join(result.raw_output for result in results)

        return ExtractionResult(
            raw_output=combined_raw,
            structured_data=normalized,
            confidence=0.9,
            model=self.model,
            provider=self.provider,
        )