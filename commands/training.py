import click
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

import arabic_reshaper
from bidi.algorithm import get_display

def smart_text(txt: str) -> str:
    txt = str(txt)
    if any('\u0600' <= c <= '\u06FF' for c in txt):
        return get_display(arabic_reshaper.reshape(txt))
    return txt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

@click.group()
def train():
    pass

@train.command()
@click.option('--csv_path', required=True)
@click.option('--embedding_path', required=True)
@click.option('--label_col', required=True)
@click.option('--test_size', default=0.2, type=float)
def models(csv_path, embedding_path, label_col, test_size):

    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    with open(embedding_path, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            X = data['vectors'].toarray()
        else:
            X = data

    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    click.echo(f'\n{"="*60}')
    click.echo('Dataset Info:')
    click.echo(f'Total samples: {len(X)}')
    click.echo(f'Train: {len(X_train)} | Test: {len(X_test)}')
    click.echo(f'Features: {X.shape[1]}')
    click.echo(f'Classes: {len(np.unique(y))}')
    click.echo(f'{"="*60}\n')

    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs('outputs/visualizations', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)

    classes = np.unique(y)
    classes_disp = [smart_text(c) for c in classes]

    for name, clf in classifiers.items():
        click.echo(f'Training {name}...')

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })

        click.echo(f'  Accuracy:  {acc:.4f}')
        click.echo(f'  Precision: {prec:.4f}')
        click.echo(f'  Recall:    {rec:.4f}')
        click.echo(f'  F1-Score:  {f1:.4f}\n')

        cm = confusion_matrix(y_test, y_pred, labels=classes)

        plt.figure(figsize=(9, 7))
        ax = sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes_disp,
            yticklabels=classes_disp
        )

        plt.tight_layout()
        plt.savefig(
            f'outputs/visualizations/cm_{name.lower().replace(" ", "")}{timestamp}.png',
            dpi=150
        )
        plt.close()

    best = max(results, key=lambda x: x['F1-Score'])

    best_model_name = best['Model']
    best_model = classifiers[best_model_name]
    best_model_path = f'outputs/models/best_model_{best_model_name.lower().replace(" ", "")}{timestamp}.pkl'
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)

    report_path = f'outputs/reports/training_report_{timestamp}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'# Training Report - {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
        f.write('## Dataset Info\n')
        f.write(f'- Total samples: {len(X)}\n')
        f.write(f'- Train/Test split: {len(X_train)}/{len(X_test)} ({int((1-test_size)*100)}/{int(test_size*100)})\n')
        f.write(f'- Classes: {len(np.unique(y))}\n')
        f.write(f'- Features: {X.shape[1]}\n\n')

        f.write('## Model Performance\n\n')
        for res in results:
            f.write(f'### {res["Model"]}\n')
            f.write(f'- Accuracy:  {res["Accuracy"]:.4f}\n')
            f.write(f'- Precision: {res["Precision"]:.4f}\n')
            f.write(f'- Recall:    {res["Recall"]:.4f}\n')
            f.write(f'- F1-Score:  {res["F1-Score"]:.4f}\n\n')

        f.write(f'### Best Model: {best["Model"]} ⭐\n')
        f.write(f'F1-Score: {best["F1-Score"]:.4f}\n\n')
        f.write(f'Best model file: {best_model_path}\n')

    click.echo(f'{"="*60}')
    click.echo(f'Best Model: {best["Model"]}')
    click.echo(f'F1-Score: {best["F1-Score"]:.4f}')
    click.echo(f'Best model saved: {best_model_path}')
    click.echo(f'\nReport saved: {report_path}')
    click.echo('Confusion matrices saved in outputs/visualizations/')
    click.echo(f'{"="*60}\n')