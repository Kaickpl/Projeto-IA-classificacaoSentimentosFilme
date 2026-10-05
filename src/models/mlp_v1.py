import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

COLUNA_ROTULO = "sentiment"


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def perda(A2, y):
    """Entropia cruzada binária."""
    eps = 1e-12
    A2 = np.clip(A2, eps, 1 - eps)
    return -np.mean(y * np.log(A2) + (1 - y) * np.log(1 - A2))


class MLP:
    """MLP com uma camada oculta, sigmoid em todos os neurônios, implementado do zero."""

    def __init__(self, n_oculta=32, eta=0.5, epocas_max=10000, paciencia=200, semente=42):
        self.n_oculta = n_oculta
        self.eta = eta
        self.epocas_max = epocas_max
        self.paciencia = paciencia
        self.semente = semente

    def _forward(self, X):
        A1 = sigmoid(X @ self.W1 + self.b1)
        A2 = sigmoid(A1 @ self.W2 + self.b2)
        return A1, A2

    def fit(self, X_train, y_train, X_val, y_val):
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        # Pesos iniciais
        rng = np.random.default_rng(self.semente)
        n_entrada = X_train.shape[1]
        self.W1 = rng.normal(0, 1 / np.sqrt(n_entrada), size=(n_entrada, self.n_oculta))
        self.b1 = np.zeros((1, self.n_oculta))
        self.W2 = rng.normal(0, 1 / np.sqrt(self.n_oculta), size=(self.n_oculta, 1))
        self.b2 = np.zeros((1, 1))
        m = X_train.shape[0]

        self.hist_treino, self.hist_val = [], []
        melhor_perda, sem_melhora = float("inf"), 0
        self.melhor_epoca = 0

        for epoca in range(1, self.epocas_max + 1):
            # Forward
            A1, A2 = self._forward(X_train)

            # Backward
            dZ2 = A2 - y_train
            dW2 = A1.T @ dZ2 / m
            db2 = np.mean(dZ2, axis=0, keepdims=True)
            dZ1 = (dZ2 @ self.W2.T) * A1 * (1 - A1)
            dW1 = X_train.T @ dZ1 / m
            db1 = np.mean(dZ1, axis=0, keepdims=True)

            # Atualização (gradiente descendente)
            self.W1 -= self.eta * dW1
            self.b1 -= self.eta * db1
            self.W2 -= self.eta * dW2
            self.b2 -= self.eta * db2

            # Validação e parada antecipada
            _, A2_val = self._forward(X_val)
            perda_val = perda(A2_val, y_val)
            self.hist_treino.append(perda(A2, y_train))
            self.hist_val.append(perda_val)

            if perda_val < melhor_perda:
                melhor_perda, sem_melhora = perda_val, 0
                self.melhor_epoca = epoca
                melhores = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
            else:
                sem_melhora += 1
                if sem_melhora >= self.paciencia:
                    break

        # Volta para os pesos do melhor momento
        self.W1, self.b1, self.W2, self.b2 = melhores
        return self

    def predict_proba(self, X):
        _, A2 = self._forward(X)
        return A2.ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def carregar_dados(caminho):
    df = pd.read_csv(caminho)
    y = df[COLUNA_ROTULO].values
    X = df.drop(columns=[COLUNA_ROTULO]).values.astype(float)
    return X, y


def treinar(X_train, y_train):
    # Separa parte do treino para validação (usada na parada antecipada)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    modelo = MLP(n_oculta=32, eta=0.5)
    modelo.fit(X_tr, y_tr, X_val, y_val)
    return modelo


if __name__ == "__main__":
    # Importa pelo caminho do pacote para o modelo salvo poder ser carregado depois
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.mlp_v1 import carregar_dados, treinar

    X, y = carregar_dados("data/processed/IMDB_menor_embeddings.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    modelo = treinar(X_train, y_train)
    print("Melhor época:", modelo.melhor_epoca)
    print(classification_report(y_test, modelo.predict(X_test)))
    joblib.dump(modelo, "models/mlp_v1.joblib")
