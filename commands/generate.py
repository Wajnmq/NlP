import click
import pandas as pd
import time
import json
import os

@click.group()
def generate():
    pass

@generate.command()
@click.option('--api_key', required=True, help='Groq API key')
@click.option('--categories', required=True, help='Comma-separated categories')
@click.option('--count', default=500, type=int, help='Total samples')
@click.option('--language', default='ar', help='Language: ar or en')
@click.option('--output', default=None, help='Output CSV path')
def data(api_key, categories, count, language, output):
    from groq import Groq
    
    cats = [c.strip() for c in categories.split(',')]
    per_category = count // len(cats)
    
    if output is None:
        counter = 1
        while os.path.exists(f'data/generated_data_{counter}.csv'):
            counter += 1
        output = f'data/generated_data_{counter}.csv'
    
    client = Groq(api_key=api_key)
    all_data = []
    
    lang_name = 'Arabic' if language == 'ar' else 'English'
    
    for idx, cat in enumerate(cats, 1):
        click.echo(f'[{idx}/{len(cats)}] Generating {per_category} samples for: {cat}')
        
        batch_size = 100
        num_batches = (per_category + batch_size - 1) // batch_size
        
        for batch_num in range(num_batches):
            current_batch = min(batch_size, per_category - batch_num * batch_size)
            
            prompt = f"""Generate exactly {current_batch} DIFFERENT {lang_name} sentences about {cat}.

STRICT RULES:
1. ALL text MUST be in {lang_name} only
2. Each sentence must be UNIQUE
3. Vary topics within {cat} category
4. NO mixing languages
5. Return ONLY a JSON array

Example: ["sentence 1", "sentence 2"]

Generate {current_batch} samples:"""

            try:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": f"You are a {lang_name} text generator. Return valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=1.2,
                    max_tokens=4000
                )
                
                text = response.choices[0].message.content.strip()
                
                if '```' in text:
                    text = text.split('```')[1]
                    if text.startswith('json'):
                        text = text[4:]
                    text = text.strip()
                
                samples = json.loads(text)
                
                for sample in samples:
                    if sample and isinstance(sample, str):
                        clean = sample.strip()
                        if len(clean) > 10:
                            all_data.append({'text': clean, 'category': cat})
                
                click.echo(f'  Batch {batch_num+1}/{num_batches}: Added samples')
                time.sleep(2)
                
            except Exception as e:
                click.echo(f'  Batch {batch_num+1} Error: {str(e)[:80]}', err=True)
                continue
        
        total_for_cat = len([d for d in all_data if d['category'] == cat])
        click.echo(f'  Total for {cat}: {total_for_cat}')
    
    if not all_data:
        click.echo('No data generated!', err=True)
        return
    
    df = pd.DataFrame(all_data)
    df.to_csv(output, index=False, encoding='utf-8-sig')
    
    click.echo(f'\n{"="*50}')
    click.echo(f'Total generated: {len(df)}')
    click.echo(f'Language: {lang_name}')
    click.echo(f'Saved: {output}')
    click.echo(f'\nDistribution:')
    for cat, cnt in df['category'].value_counts().items():
        click.echo(f'  {cat}: {cnt}')
    click.echo(f'{"="*50}')