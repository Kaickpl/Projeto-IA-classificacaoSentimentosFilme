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
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "IMDB_menor_embeddings.csv"
)

TEST_SIZE = 0.20
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


def split_data(X, y):
    """
    Divide os dados em treinamento, validação e teste.

    Proporção:
    - 70% treinamento
    - 15% validação
    - 15% teste
    """

    # 70% treinamento
    # 30% temporário (validação + teste)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Divide os 30% temporários em:
    # 15% validação
    # 15% teste
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


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
    df = load_data(INPUT_FILE)

    print("Formato do dataset:", df.shape)

    print("\nDistribuição das classes:")
    print(df["sentiment"].value_counts().sort_index())

    X, y = prepare_data(df)

    print("\nFormato de X:", X.shape)
    print("Formato de y:", y.shape)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nDados de treinamento:", X_train.shape)
    print("Dados de teste:", X_test.shape)

    print("\nDistribuição no treinamento:")
    print(y_train.value_counts().sort_index())

    print("\nDistribuição no teste:")
    print(y_test.value_counts().sort_index())

    model = create_model()

    train_model(
        model,
        X_train,
        y_train,
    )

    metrics, y_pred, y_proba = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("\n=== RESULTADOS ===")

    print(
        f"Acurácia:  {metrics['accuracy']:.4f}"
    )

    print(
        f"Precisão:  {metrics['precision']:.4f}"
    )

    print(
        f"Recall:    {metrics['recall']:.4f}"
    )

    print(
        f"F1-score:  {metrics['f1']:.4f}"
    )

    print("\nMatriz de confusão:")
    print(metrics["confusion_matrix"])

    print("\nRelatório de classificação:")
    print(metrics["classification_report"])