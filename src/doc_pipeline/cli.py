"""Command-line runner for the document pipeline."""
import click
from .ingestion import IngestionService
from .inference import InferenceService
from .storage import StorageService
from .validation import ValidationService


@click.group()
def cli():
    """Document processing pipeline CLI."""
    pass


@cli.command()
@click.argument('file_path')
@click.option('--schema', default='default', help='Extraction schema to use')
@click.option('--output-dir', default=None, help='Directory where outputs are saved')
def run(file_path: str, schema: str, output_dir: str | None):
    """Run extraction on a single document."""
    click.echo(f'📥 Loading {file_path}')
    ingestion = IngestionService()
    document = ingestion.load(file_path)

    click.echo('🤖 Extracting structured output from document')
    inference = InferenceService()
    result = inference.extract(document, schema=schema)

    click.echo('✅ Validating extraction result')
    validator = ValidationService()
    validator.validate(result.structured_data, schema=schema)

    click.echo('💾 Saving result')
    storage = StorageService(output_dir=output_dir)
    output_path = storage.save(result, document.metadata)

    click.echo(f'🎉 Saved output to {output_path}')


if __name__ == '__main__':
    cli()