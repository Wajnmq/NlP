import click
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

@click.group()
def embed():
    pass

@embed.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--max_features', default=5000, type=int)
@click.option('--output', required=True)
def tfidf(csv_path, text_col, max_features, output):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    texts = df[text_col].astype(str).tolist()
    
    click.echo(f'\nVectorizing {len(texts)} samples with TF-IDF...')
    
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(texts)
    
    with open(output, 'wb') as f:
        pickle.dump({'vectors': vectors, 'vectorizer': vectorizer}, f)
    
    click.echo(f'Shape: {vectors.shape}')
    click.echo(f'Features: {max_features}')
    click.echo(f'Saved: {output}\n')

@embed.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--output', required=True)
def bert(csv_path, text_col, output):
    from sentence_transformers import SentenceTransformer
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    texts = df[text_col].astype(str).tolist()
    
    click.echo(f'\nLoading multilingual BERT model...')
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    click.echo(f'Encoding {len(texts)} samples...')
    vectors = model.encode(texts, show_progress_bar=True)
    
    with open(output, 'wb') as f:
        pickle.dump(vectors, f)
    
    click.echo(f'Shape: {vectors.shape}')
    click.echo(f'Saved: {output}\n')

@embed.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--output', required=True)
def model2vec(csv_path, text_col, output):
    from sentence_transformers import SentenceTransformer
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    texts = df[text_col].astype(str).tolist()
    
    click.echo(f'\nLoading Model2Vec (LaBSE - multilingual)...')
    
    try:
        model = SentenceTransformer('sentence-transformers/LaBSE')
        click.echo(f'Encoding {len(texts)} samples...')
        vectors = model.encode(texts, show_progress_bar=True)
        
        with open(output, 'wb') as f:
            pickle.dump(vectors, f)
        
        click.echo(f'Shape: {vectors.shape}')
        click.echo(f'Saved: {output}\n')
        
    except Exception as e:
        click.echo(f'Error: {str(e)}', err=True)
        click.echo('Falling back to multilingual BERT...', err=True)
        
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        vectors = model.encode(texts, show_progress_bar=True)
        
        with open(output, 'wb') as f:
            pickle.dump(vectors, f)
        
        click.echo(f'Saved: {output}\n')