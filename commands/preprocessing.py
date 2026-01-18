import click
import pandas as pd
import re
import os

@click.group()
def preprocess():
    pass

def load_stopwords(language='ar'):
    file_name = f'stopwords_{language}.txt'
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def ensure_exists(path):
    if not os.path.exists(path):
        raise click.ClickException(f'File not found: {path}')

def ensure_col(df, col):
    if col not in df.columns:
        raise click.ClickException(f'Column "{col}" not found. Available: {", ".join(df.columns)}')

def filter_non_arabic_chars(text):
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r'[^\u0600-\u06FF\s،؛؟.!]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@preprocess.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--language', default='ar')
@click.option('--output', required=True)
def remove(csv_path, text_col, language, output):
    ensure_exists(csv_path)
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    ensure_col(df, text_col)

    before_sample = str(df[text_col].iloc[0])[:100] if len(df) else ""

    def clean_text(text):
        text = "" if pd.isna(text) else str(text)

        if language == 'ar':
            text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
            text = re.sub(r'ـ+', '', text)

        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        text = re.sub(r'[0-9\u0660-\u0669]+', ' ', text)

        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if language == 'ar':
            text = filter_non_arabic_chars(text)

        return text

    df[text_col] = df[text_col].apply(clean_text)

    after_sample = str(df[text_col].iloc[0])[:100] if len(df) else ""

    df.to_csv(output, index=False, encoding='utf-8-sig')
    click.echo(f'\nBefore: {before_sample}...')
    click.echo(f'After:  {after_sample}...')
    click.echo(f'\nSaved: {output}\n')

@preprocess.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--language', default='ar')
@click.option('--output', required=True)
def stopwords(csv_path, text_col, language, output):
    ensure_exists(csv_path)
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    ensure_col(df, text_col)

    stops = load_stopwords(language)

    if not stops:
        click.echo(f'Error: stopwords_{language}.txt not found', err=True)
        df.to_csv(output, index=False, encoding='utf-8-sig')
        return

    click.echo(f'Loaded {len(stops)} stopwords from stopwords_{language}.txt')

    before_sample = str(df[text_col].iloc[0])[:100] if len(df) else ""
    before_len = df[text_col].astype(str).str.split().str.len().mean()

    def remove_stops(text):
        text = "" if pd.isna(text) else str(text)
        words = text.split()
        if language == 'en':
            filtered = [w for w in words if w.lower() not in stops]
        else:
            filtered = [w for w in words if w not in stops]
        return ' '.join(filtered)

    df[text_col] = df[text_col].apply(remove_stops)

    after_sample = str(df[text_col].iloc[0])[:100] if len(df) else ""
    after_len = df[text_col].astype(str).str.split().str.len().mean()

    df.to_csv(output, index=False, encoding='utf-8-sig')
    click.echo(f'\nBefore: {before_sample}...')
    click.echo(f'After:  {after_sample}...')
    click.echo(f'Avg words: {before_len:.1f} -> {after_len:.1f}')
    click.echo(f'\nSaved: {output}\n')

@preprocess.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--language', default='ar')
@click.option('--output', required=True)
def normalize(csv_path, text_col, language, output):
    ensure_exists(csv_path)
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    ensure_col(df, text_col)

    before_sample = str(df[text_col].iloc[0])[:100] if len(df) else ""

    def normalize_text(text):
        text = "" if pd.isna(text) else str(text)
        if language == 'ar':
            text = re.sub(r'[إأآا]', 'ا', text)
            text = re.sub(r'ؤ', 'و', text)
            text = re.sub(r'ئ', 'ي', text)
            text = re.sub(r'ة', 'ه', text)
            text = re.sub(r'ى', 'ي', text)
        return text

    df[text_col] = df[text_col].apply(normalize_text)

    after_sample = str(df[text_col].iloc[0])[:100] if len(df) else ""

    df.to_csv(output, index=False, encoding='utf-8-sig')
    click.echo(f'\nBefore: {before_sample}...')
    click.echo(f'After:  {after_sample}...')
    click.echo(f'\nSaved: {output}\n')

@preprocess.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--output', required=True)
def dedup(csv_path, text_col, output):
    ensure_exists(csv_path)
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    ensure_col(df, text_col)

    before = len(df)
    duplicates = df.duplicated(subset=[text_col]).sum()

    df = df.drop_duplicates(subset=[text_col], keep='first')

    after = len(df)
    removed = before - after

    df.to_csv(output, index=False, encoding='utf-8-sig')

    click.echo(f'\n{"="*50}')
    click.echo('Duplicate Removal:')
    click.echo(f'Before: {before} samples')
    click.echo(f'Duplicates found: {duplicates}')
    click.echo(f'After: {after} samples')
    click.echo(f'Removed: {removed}')
    click.echo(f'\nSaved: {output}')
    click.echo(f'{"="*50}\n')

@preprocess.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--language', default='ar')
@click.option('--output', required=True)
def all(csv_path, text_col, language, output):
    click.echo(f'\n[1/4] Removing special chars...')
    remove.callback(csv_path, text_col, language, 'temp_step1.csv')

    click.echo('[2/4] Removing stopwords...')
    stopwords.callback('temp_step1.csv', text_col, language, 'temp_step2.csv')

    click.echo('[3/4] Normalizing text...')
    normalize.callback('temp_step2.csv', text_col, language, 'temp_step3.csv')

    click.echo('[4/4] Removing duplicates...')
    dedup.callback('temp_step3.csv', text_col, output)

    if os.path.exists('temp_step1.csv'):
        os.remove('temp_step1.csv')
    if os.path.exists('temp_step2.csv'):
        os.remove('temp_step2.csv')
    if os.path.exists('temp_step3.csv'):
        os.remove('temp_step3.csv')

    click.echo('All steps completed!')
    click.echo(f'Final output: {output}\n')
