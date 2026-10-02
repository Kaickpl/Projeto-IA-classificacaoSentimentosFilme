# src/train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

COLUNA_ROTULO = "coluna_do_rotulo"

def carregar_dados(caminho):
    df = pd.read_csv(caminho)
    y = df[COLUNA_ROTULO].values
    X = df.drop(columns=[COLUNA_ROTULO]).values.astype(float)
    return X, y

def treinar(X_train, y_train):
    modelo = KNeighborsClassifier(n_neighbors=31, weights="distance", metric="cosine")
    modelo.fit(X_train, y_train)
    return modelo

if __name__ == "__main__":
    X, y = carregar_dados("data/seu_arquivo.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    modelo = treinar(X_train, y_train)
    print(classification_report(y_test, modelo.predict(X_test)))
    joblib.dump(modelo, "models/knn_v1.joblib")