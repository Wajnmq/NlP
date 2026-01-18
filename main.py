import click
import os

@click.group()
def cli():
    os.makedirs('outputs/visualizations', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    os.makedirs('outputs/embeddings', exist_ok=True)
    os.makedirs('data', exist_ok=True)

if __name__ == '__main__':
    from commands import generate, eda, preprocessing, embedding, training
    
    cli.add_command(generate.generate)
    cli.add_command(eda.eda)
    cli.add_command(preprocessing.preprocess)
    cli.add_command(embedding.embed)
    cli.add_command(training.train)
    
    cli()