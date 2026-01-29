#!/usr/bin/env python3
import click
import json
import uuid
from pathlib import Path

ASSISTANTS_DIR = Path("resources/data/assistants")

@click.group()
def cli():
    """Tito.AI Assistant Manager CLI"""
    pass

@cli.command()
def list():
    """List all assistants"""
    for json_file in ASSISTANTS_DIR.glob("*.json"):
        if json_file.name == "migration_mapping.json":
            continue
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            click.echo(f"{data['id'][:8]}... - {data.get('name', 'Unknown')}")

@cli.command()
@click.argument('assistant_id')
def show(assistant_id):
    """Show assistant details"""
    file_path = ASSISTANTS_DIR / f"{assistant_id}.json"
    if not file_path.exists():
        click.echo(f"Assistant {assistant_id} not found", err=True)
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        click.echo(json.dumps(json.load(f), indent=2, ensure_ascii=False))

@cli.command()
@click.argument('name')
@click.option('--template', help='Copy from existing assistant UUID')
def create(name, template):
    """Create new assistant"""
    new_uuid = str(uuid.uuid4())
    
    if template:
        template_path = ASSISTANTS_DIR / f"{template}.json"
        if not template_path.exists():
            click.echo(f"Template {template} not found", err=True)
            return
        with open(template_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {"agent": {"provider": "google"}, "io_layer": {"transport": {"provider": "daily"}}}
    
    config.update({"id": new_uuid, "name": name})
    
    with open(ASSISTANTS_DIR / f"{new_uuid}.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    click.echo(f"Created: {new_uuid}")

@cli.command()
@click.argument('assistant_id')
@click.confirmation_option()
def delete(assistant_id):
    """Delete assistant"""
    file_path = ASSISTANTS_DIR / f"{assistant_id}.json"
    if not file_path.exists():
        click.echo(f"Assistant {assistant_id} not found", err=True)
        return
    
    file_path.unlink()
    click.echo(f"Deleted: {assistant_id}")

if __name__ == '__main__':
    cli()
