import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from keras_stat_tuner import StatisticalOracle, StatisticalSearch
from keras_stat_tuner.StatsTuner import ImplementationError
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras_tuner import HyperParameters, Objective
import numpy as np

train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_y = np.array([0, 1, 1, 0])


def nudge_expectations(i, s):
    base = {"units": 3, "activation": "relu", "drop": False, "lr": 1e-3}
    if i == 0:
        base["units"] = base["units"] + (s * -1)
    if i == 3:
        print(base["lr"], base["lr"], (1e-4 * s))
        base["lr"] = base["lr"] - (1e-4 * s)
    if i == 2:
        base["drop"] = True
    if i == 1:
        values = "relu,sigmoid,swish".split(",")
        base["activation"] = values[-s]
    return base


def build_hypermodel(hp: HyperParameters):
    mdl = Sequential()
    mdl.add(Dense(hp.Int("units", 1, 16, 1, default=3)))
    mdl.add(Activation(hp.Choice("activation", "relu,sigmoid,swish".split(","))))
    if hp.Boolean("drop"):
        mdl.add(Dropout(0.25))
    mdl.add(Dense(2))
    lr = hp.Float("lr", 1e-4, 1e-2, 1e-4, default=1e-3)
    mdl.compile(optimizer=Adam(learning_rate=lr),
                loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return mdl


def test_raise_error_if_no_estimator_oracle():
    with pytest.raises(ImplementationError):
        StatisticalOracle(estimator=None)


def test_raise_error_if_no_estimator_tuner():
    with pytest.raises(ImplementationError):
        StatisticalSearch(estimator=None)


def test_create_random_trials():
    tuner = StatisticalSearch(
        hypermodel=build_hypermodel,
        estimator=Pipeline([("model", LinearRegression())]),
        objective=Objective("accuracy", "max"),
        initial_trials=3,
        max_trials=3
    )
    tuner.search(train_x, train_y, epochs=1, verbose=0)


def test_create_nudged_trials():
    project_name = "test_nudge_create"
    tuner = StatisticalSearch(
        hypermodel=build_hypermodel,
        estimator=Pipeline([("model", LinearRegression())]),
        objective=Objective("accuracy", "max"),
        initial_trials=3,
        max_trials=5,
        project_name=project_name
    )
    tuner.search(train_x, train_y, epochs=1, verbose=0)


@pytest.mark.parametrize(
    argnames=["index", "sign", "expected"],
    argvalues=[(i, s, nudge_expectations(i, s)) for i in range(4) for s in [-1, 1]],
    ids=["intnudgeup", "intnudgedown", "choicenudgeup", "choicenudgedown",
         "boolnudgeup",
         "boolnudgedown", "floatnudgeup", "floatnudgedown"]
)
def test_specific_nudge(index, sign, expected):
    tuner = StatisticalSearch(
        hypermodel=build_hypermodel,
        estimator=Pipeline([("model", LinearRegression())]),
        objective=Objective("accuracy", "max"),
        initial_trials=3,
        max_trials=3,
        project_name="nudge"
    )
    oracle = tuner.oracle
    hp = oracle.hyperparameters
    nudged = oracle.nudge(hp, index, sign)
    assert nudged == expected

def test_no_nudge():
    tuner = StatisticalSearch(
        hypermodel=build_hypermodel,
        estimator=Pipeline([("model", LinearRegression())]),
        objective=Objective("accuracy", "max"),
        initial_trials=3,
        max_trials=3,
        project_name="nudge"
    )
    oracle = tuner.oracle
    hp = oracle.hyperparameters
    nudged = oracle.nudge(hp, 0, 0)
    assert nudged != hp.values
