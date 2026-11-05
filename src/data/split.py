
import math

from sklearn.model_selection import train_test_split


TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42


def split_data(X, y):

    total = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

    if not math.isclose(total, 1.0):
        raise ValueError(
            "TRAIN_SIZE, VAL_SIZE e TEST_SIZE precisam somar 1. "
            f"Soma atual: {total}"
        )

    n_samples = len(y)
    n_val = round(n_samples * VAL_SIZE)
    n_test = round(n_samples * TEST_SIZE)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=n_val + n_test,
        random_state=RANDOM_STATE,
        stratify=y,
    )


    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=n_test,
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
