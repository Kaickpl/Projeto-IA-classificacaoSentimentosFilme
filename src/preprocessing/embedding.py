"""Embedding das avaliações (coluna ``review``)."""
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "IMDB_Dataset_menor_encoded.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "IMDB_menor_embeddings.npy"
)

CSV_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "IMDB_menor_embeddings.csv"
)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 64

## ta carregando a tabela que npo caso é o encodding
def load_model(model_name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def check_reviews(df: pd.DataFrame) -> None:
    if "review" not in df.columns:
        raise ValueError(
            "A coluna 'review' não foi encontrada no dataset."
        )

    missing = df["review"].isnull().sum()
    empty = (df["review"].astype(str).str.strip() == "").sum()

    if missing or empty:
        raise ValueError(
            f"A coluna 'review' possui {missing} valores ausentes "
            f"e {empty} valores vazios."
        )


def generate_embeddings(
    texts: list[str],
    model=None,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    if model is None:
        model = load_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return np.asarray(embeddings, dtype=np.float32)


def validate_embeddings(embeddings: np.ndarray, n_reviews: int) -> None:
    expected_shape = (n_reviews, EMBEDDING_DIM)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Formato inesperado: {embeddings.shape}, "
            f"esperado {expected_shape}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Os embeddings possuem valores inválidos (NaN ou infinito)."
        )


def embeddings_to_dataframe(
    embeddings: np.ndarray,
    sentiments=None,
) -> pd.DataFrame:
    df_embeddings = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i}" for i in range(embeddings.shape[1])],
    )

    if sentiments is not None:
        df_embeddings["sentiment"] = np.asarray(sentiments)

    return df_embeddings


def embed_dataset(
    input_file: Path,
    output_file: Path,
    model=None,
    csv_output_file: Path | None = None,
) -> np.ndarray:

    df = pd.read_csv(input_file)

    check_reviews(df)

    # A coluna 'review' não é alterada; apenas lida para gerar os vetores
    embeddings = generate_embeddings(df["review"].tolist(), model=model)

    validate_embeddings(embeddings, len(df))

    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_file, embeddings)

    if csv_output_file is not None:
        sentiments = df["sentiment"] if "sentiment" in df.columns else None
        embeddings_to_dataframe(embeddings, sentiments).to_csv(
            csv_output_file,
            index=False,
        )

    return embeddings


if __name__ == "__main__":
    embeddings = embed_dataset(
        INPUT_FILE,
        OUTPUT_FILE,
        csv_output_file=CSV_OUTPUT_FILE,
    )

    print("Embedding concluído.")
    print(f"Arquivo salvo em: {OUTPUT_FILE}")
    print(f"Tabela CSV salva em: {CSV_OUTPUT_FILE}")
    print(f"Formato da matriz: {embeddings.shape}")