from pathlib import Path
import re
from html import unescape
import unicodedata
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IMDB_Dataset_menor.csv"  # Ajuste para o caminho do seu CSV bruto
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "IMDB_Dataset_menor_normalizado_v2.csv"
)


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1. Decodifica entidades HTML (&amp;, &quot;, etc.)
    text = unescape(text)

    # 2. Normaliza caracteres Unicode (aspas elegantes, travessões, acentos)
    text = unicodedata.normalize("NFKC", text)

    # 3. Substitui tags HTML por um espaço para não grudar palavras
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Mantém maiúsculas (linha text.lower() removida)
    # 5. Mantém pontuações (!, ?, etc.)

    # 6. Limpa múltiplos espaços e quebras de linha
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_dataset(input_file: Path, output_file: Path) -> pd.DataFrame:
    df = pd.read_csv(input_file)

    if "review" not in df.columns:
        raise ValueError("A coluna 'review' não foi encontrada no dataset.")

    df["review"] = df["review"].apply(normalize_text)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return df


if __name__ == "__main__":
    texto_teste = "I REALLY loved this movie!!! <br /><br />It was AMAZING &amp; mind-blowing."
    print("Teste isolado da função:")
    print("Original:   ", texto_teste)
    print("Normalizado:", normalize_text(texto_teste))

    if INPUT_FILE.exists():
        df_norm = normalize_dataset(INPUT_FILE, OUTPUT_FILE)
        print("\nDataset normalizado v2 gerado com sucesso!")
        print(f"Salvo em: {OUTPUT_FILE}")