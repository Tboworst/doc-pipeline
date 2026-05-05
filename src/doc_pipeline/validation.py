"""Validation service for schema checking extracted output."""
from typing import Any, Dict


class ValidationService:
    """Validate structured extraction results against simple schema rules."""

    SCHEMA_RULES: Dict[str, Dict[str, bool]] = {
        'default': {
            'title': True,
            'summary': True,
            'entities': False,
            'topics': False,
        },
        'invoice': {
            'invoice_number': True,
            'date': True,
            'vendor': True,
            'total': True,
            'line_items': False,
        },
        'contract': {
            'parties': True,
            'effective_date': True,
            'term': False,
            'obligations': False,
            'termination': False,
        },
    }

    def validate(self, data: dict, schema: str = 'default') -> dict:
        """Validate the output data against the chosen schema."""
        rules = self.SCHEMA_RULES.get(schema, self.SCHEMA_RULES['default'])
        errors = []

        for field, required in rules.items():
            value = data.get(field)
            if required and not value:
                errors.append(f"Missing required field: {field}")

        if errors:
            raise ValueError('Validation failed: ' + '; '.join(errors))

        return data
