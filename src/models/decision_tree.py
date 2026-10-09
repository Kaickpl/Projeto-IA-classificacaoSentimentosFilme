from pathlib import Path
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "IMDB_Dataset_menor.csv"

def clean_text(text: str) -> str:
    """Função para limpar HTML e caracteres especiais das avaliações."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<br\s*/?>', ' ', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    return text.lower().strip()

def roda_arvore_de_decisão():
    if not INPUT_FILE.exists():
        print(f"Erro: Arquivo não encontrado em: {INPUT_FILE}")
        print("Verifique se o arquivo IMDB_Dataset_menor_2.csv está na pasta data/raw/")
        return

    print("Carregando os dados....")
    df = pd.read_csv(INPUT_FILE)

    if 'sentiment' not in df.columns or 'review' not in df.columns:
        raise KeyError("O arquivo CSV precisa conter as colunas 'review' e 'sentiment'.")

    print("Limpando o texto das avaliações...")
    df['clean_review'] = df['review'].apply(clean_text)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        df['clean_review'], 
        df['sentiment'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['sentiment']
    )

    print("Vetorizando o texto com TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')
    x_train = vectorizer.fit_transform(x_train_raw)
    x_test = vectorizer.transform(x_test_raw)

    print(f"Formato da matriz de características (X): {x_train.shape}")

    print("Treinando a árvore de decisão...")
    clf = DecisionTreeClassifier(max_depth=15, min_samples_split=10, random_state=42)
    clf.fit(x_train, y_train)

    print("\n================ RELATÓRIO DE CLASSIFICAÇÃO ================")
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print("============================================================\n")

    # salva o gráfico da matriz de confusão na pasta results/figures
    output_dir = PROJECT_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    plt.tight_layout()

    matrix_path = output_dir / "arvore_de_decisao_cm.png"
    plt.savefig(matrix_path)
    print(f"Gráfico da matriz de confusão salvo em: {matrix_path}")

if __name__ == "__main__":
    roda_arvore_de_decisão()