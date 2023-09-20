from keras_stat_tuner import StatisticalOracle, StatisticalSearch
from keras_stat_tuner.StatsTuner import ImplementationError
import pytest

def test_raise_error_if_no_estimator_oracle():
    with pytest.raises(ImplementationError):
        StatisticalOracle(estimator=None)

def test_raise_error_if_no_estimator_tuner():
    with pytest.raises(ImplementationError):
        StatisticalSearch(estimator=None)

