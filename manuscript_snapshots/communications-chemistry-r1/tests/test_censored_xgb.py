import numpy as np

from canal.censored_xgb import censored_grad_hess


def test_censored_gradient_direction_left_bound():
    pred = np.array([7.0])
    y = np.array([6.0])
    rel = np.array(["<"])
    grad, hess = censored_grad_hess(pred, y, rel, sigma=1.0)
    assert grad[0] > 0
    assert hess[0] > 0


def test_censored_gradient_direction_right_bound():
    pred = np.array([5.0])
    y = np.array([6.0])
    rel = np.array([">"])
    grad, hess = censored_grad_hess(pred, y, rel, sigma=1.0)
    assert grad[0] < 0
    assert hess[0] > 0


def test_censor_weight_scales_censored_gradient():
    pred = np.array([5.0])
    y = np.array([6.0])
    rel = np.array([">"])
    grad_full, _ = censored_grad_hess(pred, y, rel, sigma=1.0, censor_weight=1.0)
    grad_half, _ = censored_grad_hess(pred, y, rel, sigma=1.0, censor_weight=0.5)
    assert abs(grad_half[0] / grad_full[0] - 0.5) < 1e-6


def test_right_censor_weight_overrides_global_weight():
    pred = np.array([5.0])
    y = np.array([6.0])
    rel = np.array([">"])
    grad_global, _ = censored_grad_hess(pred, y, rel, sigma=1.0, censor_weight=1.0)
    grad_right, _ = censored_grad_hess(
        pred,
        y,
        rel,
        sigma=1.0,
        censor_weight=1.0,
        right_censor_weight=0.25,
    )
    assert abs(grad_right[0] / grad_global[0] - 0.25) < 1e-6


def test_censored_anchor_adds_point_gradient():
    pred = np.array([8.0])
    y = np.array([6.0])
    rel = np.array(["<"])
    grad_plain, hess_plain = censored_grad_hess(pred, y, rel, sigma=1.0, censor_weight=0.5)
    grad_anchor, hess_anchor = censored_grad_hess(
        pred,
        y,
        rel,
        sigma=1.0,
        censor_weight=0.5,
        censored_anchor_weight=0.25,
    )
    assert grad_anchor[0] > grad_plain[0]
    assert hess_anchor[0] > hess_plain[0]


def test_right_anchor_can_differ_from_left_anchor():
    pred = np.array([8.0, 5.0])
    y = np.array([6.0, 6.0])
    rel = np.array(["<", ">"])
    grad_plain, hess_plain = censored_grad_hess(
        pred,
        y,
        rel,
        sigma=1.0,
        censor_weight=0.5,
        censored_anchor_weight=0.0,
    )
    grad_right_anchor, hess_right_anchor = censored_grad_hess(
        pred,
        y,
        rel,
        sigma=1.0,
        censor_weight=0.5,
        censored_anchor_weight=0.0,
        right_censored_anchor_weight=0.25,
    )
    assert grad_right_anchor[0] == grad_plain[0]
    assert hess_right_anchor[0] == hess_plain[0]
    assert grad_right_anchor[1] < grad_plain[1]
    assert hess_right_anchor[1] > hess_plain[1]
