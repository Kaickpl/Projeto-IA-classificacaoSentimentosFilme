from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "IMDB_Dataset_menor_normalizado_v2.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "IMDB_Dataset_menor_encoded_v2.csv"
)


def encode_sentiment(sentiment: str) -> int:
    mapping = {
        "negative": 0,
        "positive": 1,
    }

    if sentiment not in mapping:
        raise ValueError(f"Sentimento inválido encontrado: {sentiment!r}")

    return mapping[sentiment]


def encode_dataset(input_file: Path, output_file: Path) -> pd.DataFrame:
    df = pd.read_csv(input_file)

    if "sentiment" not in df.columns:
        raise ValueError("A coluna 'sentiment' não foi encontrada no dataset.")

    sentiments = set(df["sentiment"].dropna().unique())
    allowed_sentiments = {"negative", "positive"}

    if not sentiments.issubset(allowed_sentiments):
        invalid = sentiments - allowed_sentiments
        raise ValueError(f"Valores inválidos na coluna 'sentiment': {invalid}")

    df["sentiment"] = df["sentiment"].apply(encode_sentiment)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return df


if __name__ == "__main__":
    if INPUT_FILE.exists():
        df_encoded = encode_dataset(INPUT_FILE, OUTPUT_FILE)
        print("Encoding v2 concluído.")
        print(f"Arquivo salvo em: {OUTPUT_FILE}")
        print("\nDistribuição dos sentimentos:")
        print(df_encoded["sentiment"].value_counts().sort_index())
    else:
        print(f"Arquivo de entrada não encontrado: {INPUT_FILE}")