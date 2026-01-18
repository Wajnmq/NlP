
# Arabic & English NLP Text Classification CLI Tool

This project is a command-line interface (CLI) tool that implements a complete Natural Language Processing pipeline for text classification.

It supports Arabic and English text and covers the full workflow starting from data generation, exploratory data analysis, preprocessing, embedding, and ending with model training and evaluation.



---

## Data Generation

Text data is generated using the LLaMA-3.3-70B model via the Groq API.

The generation process allows control over:
- Category names
- Number of samples
- Language

A valid Groq API key is required to run this step.

### Example Commands

**Arabic topic classification:**
```bash
python main.py generate data --api_key YOUR_GROQ_KEY --categories "رياضة,اقتصاد,سياسة,تقنية,صحة" --count 1500 --language ar
```

**English classification example:**
```bash
python main.py generate data --api_key YOUR_GROQ_KEY --categories "sports,economy,politics,technology,health" --count 1500 --language en
```

---

## Exploratory Data Analysis (EDA)

EDA provides a general overview of the dataset.

- **Distribution**: shows class balance
- **Histogram**: analyzes text length distribution
- **Word Cloud**: highlights frequent terms per dataset or per class

### EDA Commands

**Class distribution:**
```bash
python main.py eda distribution --csv_path data/data.csv --label_col category
```

**Text length histogram:**
```bash
python main.py eda histogram --csv_path data/data.csv --text_col text --unit words
```

**Word cloud visualization:**
```bash
python main.py eda wordcloud --csv_path data/data.csv --text_col text --label_col category --language ar
```

---

## Preprocessing

**Text preprocessing steps:**

- **Remove stopwords**  
  Uses external stopword files per language (stopwords_ar.txt, stopwords_en.txt)

- **Remove characters**  
  Removes diacritics, URLs, numbers, symbols, and non-relevant characters

- **Normalize text**  
  Arabic normalization rules:  
  إ, أ, آ → ا  
  ة → ه  
  ؤ → و  
  ئ → ي  
  ى → ي

- **Remove duplicates**  
  Removes duplicated text samples

Each step can be executed individually or all steps can be applied in a single command.

### Step-by-Step Commands

**Step 1: Remove special characters**
```bash
python main.py preprocess remove --csv_path data.csv --text_col text --language ar --output step1.csv
```

**Step 2: Remove stopwords**
```bash
python main.py preprocess stopwords --csv_path step1.csv --text_col text --language ar --output step2.csv
```

**Step 3: Normalize text**
```bash
python main.py preprocess normalize --csv_path step2.csv --text_col text --language ar --output step3.csv
```

**Step 4: Remove duplicates**
```bash
python main.py preprocess dedup --csv_path step3.csv --text_col text --output final.csv
```

### Full Preprocessing Command


```bash
python main.py preprocess all --csv_path data.csv --text_col text --language ar --output final.csv
```

---

## Text Embedding

The project supports the following embedding methods:

- **TF-IDF** 
- **Multilingual BERT**  
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Model2Vec (LaBSE)**  
  sentence-transformers/LaBSE

### Embedding Commands

**TF-IDF embedding:**
```bash
python main.py embed tfidf --csv_path final.csv --text_col text --max_features 5000 --output tfidf.pkl
```

**BERT embedding:**
```bash
python main.py embed bert --csv_path final.csv --text_col text --output bert.pkl
```

**Model2Vec embedding:**
```bash
python main.py embed model2vec --csv_path final.csv --text_col text --output model2vec.pkl
```

---

## Training

The following models are used:

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Random Forest**

For each model, the following metrics are computed:

- Accuracy  
- Precision  
- Recall  
- F1-score  

The best model is automatically selected based on F1-score and saved.

### Training Command Example

```bash
python main.py train models --csv_path final_ar.csv --embedding_path embeddings_ar_bert.pkl --label_col category --test_size 0.3
```

---

## Example Experiment Results

These results represent an experimental run on an Arabic text classification dataset with five categories:
Sports, Economy, Politics, Technology, and Health.

**Dataset Information:**
- Total samples: 1450
- Number of classes: 5  
- Test split: 30%

**Model Results:**
- KNN: F1 ≈ 0.89  
- Logistic Regression: F1 ≈ 0.93  
- Random Forest: F1 ≈ 0.91  

The best performing model in this experiment was **Logistic Regression**.

**Confusion Matrix:**

![Confusion Matrix](outputs/visualizations/cm_randomforest20260118_060003.png)