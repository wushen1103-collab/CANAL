import numpy as np

from canal.tobit import train_linear_tobit


def test_linear_tobit_moves_left_censored_prediction_down():
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=float)
    y = np.array([1.0, 2.0, 3.0, 2.5, 2.5, 2.5], dtype=float)
    rel = np.array(["=", "=", "=", "<", "<", "<"])

    model = train_linear_tobit(x, y, rel, sigma=0.5, ridge_alpha=0.1, max_iter=80)
    pred = model.predict(np.array([[5.0]], dtype=float))

    assert pred[0] < 3.5


def test_linear_tobit_moves_right_censored_prediction_up():
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=float)
    y = np.array([5.0, 4.0, 3.0, 3.5, 3.5, 3.5], dtype=float)
    rel = np.array(["=", "=", "=", ">", ">", ">"])

    model = train_linear_tobit(x, y, rel, sigma=0.5, ridge_alpha=0.1, max_iter=80)
    pred = model.predict(np.array([[5.0]], dtype=float))

    assert pred[0] > 2.5
