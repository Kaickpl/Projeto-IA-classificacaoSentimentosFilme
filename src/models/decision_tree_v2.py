import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Permite importar os módulos de src/
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.split import split_data  # noqa: E402

# Utiliza os embeddings gerados pela nova normalização v2
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "IMDB_menor_embeddings_v2.csv"
COLUNA_ROTULO = "sentiment"
RANDOM_STATE = 42

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
    resultados = []
    melhores_params, melhor_f1 = None, -1.0

    for params in ParameterGrid(GRADE):
        modelo = treinar(X_train, y_train, **params)
        f1_val = f1_score(y_val, modelo.predict(X_val), average="macro")

        resultados.append({
            **{f"param_{nome}": valor for nome, valor in params.items()},
            "val_f1_macro": f1_val,
        })

        if f1_val > melhor_f1:
            melhores_params, melhor_f1 = params, f1_val

    resultados = (
        pd.DataFrame(resultados)
        .sort_values("val_f1_macro", ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    best_dt = treinar(X_train, y_train, **melhores_params)
    return best_dt, melhores_params, resultados


def roda_arvore_de_decisao():
    if not INPUT_FILE.exists():
        print(f"Erro: Arquivo não encontrado em: {INPUT_FILE}")
        print("Gere os embeddings v2 (src/preprocessing/embedding_v2.py) antes de rodar este arquivo.")
        return

    print("Carregando embeddings v2...")
    X, y = carregar_dados(INPUT_FILE)
    print(f"Formato da matriz de características (X): {X.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("Buscando hiperparâmetros na validação...")
    # Ajustado para receber os 3 valores retornados por buscar_hiperparametros
    best_dt, melhores_params, resultados_busca = buscar_hiperparametros(X_train, y_train, X_val, y_val)
    print(f"Melhores parâmetros encontrados: {melhores_params}")

    # Treinamento final com Treino + Validação
    X_train_final = np.vstack((X_train, X_val))
    y_train_final = np.hstack((y_train, y_val))
    clf = treinar(X_train_final, y_train_final, **melhores_params)

    print("\n================ RELATÓRIO DE CLASSIFICAÇÃO (TESTE V2) ================")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("=======================================================================\n")

    output_dir = PROJECT_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    plt.title("Matriz de Confusão - Árvore de Decisão (v2)")
    plt.tight_layout()

    matrix_path = output_dir / "arvore_de_decisao_v2_cm.png"
    plt.savefig(matrix_path)
    print(f"Gráfico da matriz de confusão salvo em: {matrix_path}")


if __name__ == "__main__":
    roda_arvore_de_decisao()