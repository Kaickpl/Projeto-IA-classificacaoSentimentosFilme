import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.split import split_data  # noqa: E402

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "IMDB_menor_embeddings_v2.csv"
COLUNA_ROTULO = "sentiment"

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
    return melhores_params, resultados


if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"Erro: Arquivo não encontrado em: {INPUT_FILE}")
        sys.exit(1)

    X, y = carregar_dados(INPUT_FILE)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("Buscando hiperparâmetros para KNN (v2)...")
    melhores_params, _ = buscar_hiperparametros(X_train, y_train, X_val, y_val)
    print("Melhores parâmetros encontrados:", melhores_params)

    # Treinamento final unindo Treino + Validação
    X_train_final = np.vstack((X_train, X_val))
    y_train_final = np.hstack((y_train, y_val))
    modelo = treinar(X_train_final, y_train_final, **melhores_params)

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO KNN (TESTE V2) ===")
    print(classification_report(y_test, modelo.predict(X_test)))

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, models_dir / "knn_v2.joblib")
    print(f"Modelo salvo em: {models_dir / 'knn_v2.joblib'}")