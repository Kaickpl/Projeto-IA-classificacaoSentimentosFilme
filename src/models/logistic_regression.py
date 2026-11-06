import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Permite importar os módulos de src/ também ao executar este arquivo diretamente
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# A divisão em treinamento, validação e teste é a mesma para todos os modelos.
# As proporções são definidas em src/data/split.py.
from data.split import split_data  # noqa: E402

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "IMDB_menor_embeddings.csv"
)

RANDOM_STATE = 42
MAX_ITER = 1000


def load_data(input_file: Path) -> pd.DataFrame:
    """Carrega e valida os dados dos embeddings"""

    df = pd.read_csv(input_file)

    if "sentiment" not in df.columns:
        raise ValueError(
            "A coluna 'sentiment' não foi encontrada."
        )

    embedding_columns = [
        column
        for column in df.columns
        if column.startswith("emb_")
    ]

    if not embedding_columns:
        raise ValueError(
            "Nenhuma coluna de embedding foi encontrada."
        )

    if df[embedding_columns + ["sentiment"]].isnull().any().any():
        raise ValueError(
            "Foram encontrados valores ausentes nos embeddings ou no sentimento."
        )

    valid_labels = {0, 1}
    labels = set(df["sentiment"].unique())

    if not labels.issubset(valid_labels):
        raise ValueError(
            f"Rótulos inválidos encontrados: {labels - valid_labels}"
        )

    return df


def prepare_data(df: pd.DataFrame):
    """Separa características e rótulos"""

    embedding_columns = [
        column
        for column in df.columns
        if column.startswith("emb_")
    ]

    X = df[embedding_columns]
    y = df["sentiment"]

    return X, y


def create_model(C: float = 1.0) -> LogisticRegression:
    """
    Cria o modelo de Regressão Logística.

    O parâmetro C controla a regularização.
    """

    return LogisticRegression(
        C=C,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )


def train_model(model, X_train, y_train):
    """Treina o modelo."""

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Avalia o modelo."""

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
        ),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
        ),
    }

    return metrics, y_pred, y_proba


if __name__ == "__main__":
    import numpy as np

    df = load_data(INPUT_FILE)

    print("Formato do dataset:", df.shape)
    print("\nDistribuição das classes:")
    print(df["sentiment"].value_counts().sort_index())

    X, y = prepare_data(df)

    print("\nFormato de X:", X.shape)
    print("Formato de y:", y.shape)

    # 1. Divisão em 3 conjuntos
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("\nTreinamento:", X_train.shape)
    print("Validação:", X_val.shape)
    print("Teste:", X_test.shape)

    # 2. Busca do melhor hiperparâmetro C no conjunto de validação
    C_VALUES = np.logspace(-2, 2, 13)
    best_f1 = -1.0
    best_C = 1.0

    for C in C_VALUES:
        model = create_model(C=C)
        train_model(model, X_train, y_train)
        metrics, _, _ = evaluate_model(model, X_val, y_val)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_C = C

    print(f"\nMelhor C encontrado na validação: {best_C} (F1-score: {best_f1:.4f})")

    # 3. Treinamento do modelo final (Treino + Validação)
    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = pd.concat([y_train, y_val], axis=0)

    final_model = create_model(C=float(best_C))
    train_model(final_model, X_train_final, y_train_final)

    # 4. Avaliação final no conjunto de teste
    test_metrics, y_pred, y_proba = evaluate_model(final_model, X_test, y_test)

    print("\n=== RESULTADO FINAL NO TESTE ===")
    print(f"Acurácia:  {test_metrics['accuracy']:.4f}")
    print(f"Precisão:  {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-score:  {test_metrics['f1']:.4f}")

    print("\nMatriz de confusão:")
    print(test_metrics["confusion_matrix"])

    print("\nRelatório de classificação:")
    print(test_metrics["classification_report"])