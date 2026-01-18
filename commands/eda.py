import click
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud

@click.group()
def eda():
    pass

@eda.command()
@click.option('--csv_path', required=True)
@click.option('--label_col', required=True)
@click.option('--plot_type', default='bar')
def distribution(csv_path, label_col, plot_type):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    counts = df[label_col].value_counts()
    
    duplicates = df.duplicated(subset=['text']).sum()
    unique = len(df) - duplicates
    
    click.echo(f'\n{"="*50}')
    click.echo(f'Dataset Overview:')
    click.echo(f'Total samples: {len(df)}')
    click.echo(f'Unique samples: {unique}')
    click.echo(f'Duplicates: {duplicates}')
    click.echo(f'Classes: {len(counts)}')
    click.echo(f'{"="*50}')
    click.echo(f'\nClass Distribution:')
    for cls, cnt in counts.items():
        click.echo(f'{cls}: {cnt} ({cnt/len(df)*100:.1f}%)')
    
    plt.figure(figsize=(8, 5))
    if plot_type == 'pie':
        plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
    else:
        plt.bar(counts.index, counts.values)
        plt.xticks(rotation=45)
    
    out = f'outputs/visualizations/dist_{plot_type}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    click.echo(f'\nSaved: {out}')
    click.echo(f'{"="*50}\n')

@eda.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--unit', default='words')
def histogram(csv_path, text_col, unit):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    if unit == 'words':
        lengths = df[text_col].str.split().str.len()
    else:
        lengths = df[text_col].str.len()
    
    click.echo(f'\nMean: {lengths.mean():.1f}')
    click.echo(f'Median: {lengths.median():.1f}')
    click.echo(f'Min: {lengths.min()} | Max: {lengths.max()}')
    
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=20, edgecolor='black')
    plt.axvline(lengths.mean(), color='red', linestyle='--', label='Mean')
    plt.xlabel(f'Length ({unit})')
    plt.ylabel('Count')
    plt.legend()
    
    out = f'outputs/visualizations/length_{unit}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    click.echo(f'\nSaved: {out}\n')

@eda.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--label_col', default=None)
@click.option('--language', default='ar')
def wordcloud(csv_path, text_col, label_col, language):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    font_path = r"C:\Windows\Fonts\arial.ttf"

    def build_wc(text, title, out_path):
        if language == 'ar':
            import arabic_reshaper
            from bidi.algorithm import get_display
            reshaped = arabic_reshaper.reshape(text)
            display_text = get_display(reshaped)
        else:
            display_text = text

        wc = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            font_path=font_path,
            collocations=False
        ).generate(display_text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    if label_col and label_col in df.columns:
        for cat in df[label_col].unique():
            text = ' '.join(df[df[label_col] == cat][text_col].astype(str))
            out_path = f'outputs/visualizations/wordcloud_{cat}.png'
            build_wc(text, f'Word Cloud - {cat}', out_path)
            click.echo(f'Saved: {out_path}')
    else:
        text = ' '.join(df[text_col].astype(str))
        out_path = 'outputs/visualizations/wordcloud_all.png'
        build_wc(text, 'Word Cloud', out_path)
        click.echo(f'Saved: {out_path}')