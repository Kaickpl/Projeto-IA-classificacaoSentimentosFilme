import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Permite importar os módulos de src/ também ao executar este arquivo diretamente
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# A divisão em treinamento, validação e teste é a mesma para todos os modelos.
# As proporções são definidas em src/data/split.py.
from data.split import split_data  # noqa: E402

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "IMDB_menor_embeddings.csv"
COLUNA_ROTULO = "sentiment"

# Combinações de hiperparâmetros testadas na busca
GRADE = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 71, 101],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "cosine"],
}


def carregar_dados(caminho):
    df = pd.read_csv(caminho)
    y = df[COLUNA_ROTULO].values
    X = df.drop(columns=[COLUNA_ROTULO]).values.astype(float)
    return X, y


def treinar(X_train, y_train, n_neighbors=31, weights="distance", metric="cosine"):
    modelo = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
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


if __name__ == "__main__":
    X, y = carregar_dados(INPUT_FILE)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    modelo, melhores_params, _ = buscar_hiperparametros(X_train, y_train, X_val, y_val)
    print("Melhores parâmetros:", melhores_params)

    print(classification_report(y_test, modelo.predict(X_test)))
    joblib.dump(modelo, "models/knn_v1.joblib")
