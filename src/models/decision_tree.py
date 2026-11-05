import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Permite importar os módulos de src/ também ao executar este arquivo diretamente
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# A divisão em treinamento, validação e teste é a mesma para todos os modelos.
# As proporções são definidas em src/data/split.py.
from data.split import split_data  # noqa: E402

# Mesmos embeddings usados pelos outros modelos (gerados em notebooks/03_embedding.ipynb)
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "IMDB_menor_embeddings.csv"
COLUNA_ROTULO = "sentiment"
RANDOM_STATE = 42

# Combinações de hiperparâmetros testadas na busca
GRADE = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"],
}


def carregar_dados(caminho):
    df = pd.read_csv(caminho)

    if COLUNA_ROTULO not in df.columns:
        raise KeyError(f"O arquivo CSV precisa conter a coluna '{COLUNA_ROTULO}'.")

    y = df[COLUNA_ROTULO].values
    X = df.drop(columns=[COLUNA_ROTULO]).values.astype(float)
    return X, y


def treinar(X_train, y_train, max_depth=None, min_samples_split=2, criterion="gini"):
    modelo = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=RANDOM_STATE,
    )
    modelo.fit(X_train, y_train)
    return modelo


def buscar_hiperparametros(X_train, y_train, X_val, y_val):
    """
    Treina uma combinação da GRADE por vez no conjunto de treinamento e
    avalia no conjunto de validação (F1 macro). O teste não é usado aqui.

    Retorna o melhor modelo, os melhores parâmetros e a tabela com todas
    as combinações testadas.
    """
    resultados = []
    melhor_modelo, melhores_params, melhor_f1 = None, None, -1.0

    for params in ParameterGrid(GRADE):
        modelo = treinar(X_train, y_train, **params)
        f1_val = f1_score(y_val, modelo.predict(X_val), average="macro")

        resultados.append({
            **{f"param_{nome}": valor for nome, valor in params.items()},
            "val_f1_macro": f1_val,
        })

        if f1_val > melhor_f1:
            melhor_modelo, melhores_params, melhor_f1 = modelo, params, f1_val

    resultados = (
        pd.DataFrame(resultados)
        .sort_values("val_f1_macro", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return melhor_modelo, melhores_params, resultados


def roda_arvore_de_decisão():
    if not INPUT_FILE.exists():
        print(f"Erro: Arquivo não encontrado em: {INPUT_FILE}")
        print("Gere os embeddings com notebooks/03_embedding.ipynb antes de rodar este arquivo.")
        return

    print("Carregando os embeddings....")
    X, y = carregar_dados(INPUT_FILE)
    print(f"Formato da matriz de características (X): {X.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("Treinando a árvore de decisão e escolhendo os hiperparâmetros na validação...")
    clf, melhores_params, _ = buscar_hiperparametros(X_train, y_train, X_val, y_val)
    print(f"Melhores parâmetros: {melhores_params}")

    print("\n================ RELATÓRIO DE CLASSIFICAÇÃO ================")
    y_pred = clf.predict(X_test)
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
