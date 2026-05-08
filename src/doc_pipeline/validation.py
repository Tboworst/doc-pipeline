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
        """Validate the output data against the chosen schema.

        Missing required fields and unknown extra fields are collected into
        a 'misc' key rather than rejecting the document outright.
        """
        rules = self.SCHEMA_RULES.get(schema, self.SCHEMA_RULES['default'])
        known_fields = set(rules.keys())
        misc = {}

        # Move missing/empty required fields into misc
        for field, required in rules.items():
            if required and not data.get(field):
                misc[field] = data.pop(field, None)

        # Move any extra fields the LLM returned that aren't in the schema into misc
        for field in list(data.keys()):
            if field not in known_fields:
                misc[field] = data.pop(field)

        if misc:
            data['misc'] = misc

        return data
