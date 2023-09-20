import copy
import numpy as np
from keras_tuner import Oracle, Tuner, HyperParameters
from keras_tuner.engine.hyperparameters import Int, Choice, Boolean, Float
from keras_tuner.engine.trial import TrialStatus
from sklearn.pipeline import Pipeline


class ImplementationError(ValueError):
    pass


class StatisticalOracle(Oracle):
    """Oracle that uses a statistical model to optimise hpyerparameters.

        Args:
            objective: A string, `keras_tuner.Objective` instance, or a list of
                `keras_tuner.Objective`s and strings. If a string, the direction of
                the optimization (min or max) will be inferred. If a list of
                `keras_tuner.Objective`, we will minimize the sum of all the
                objectives to minimize subtracting the sum of all the objectives to
                maximize. The `objective` argument is optional when
                `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
                the objective to minimize.
            initial_trials: Integer, the initial number of random trials to perform before tuning. Defaults to 10
            max_trials: Integer, the total number of trials (model configurations)
                to test at most. Note that the oracle may interrupt the search
                before `max_trial` models have been tested if the search space has
                been exhausted. Defaults to 100.
            seed: Optional integer, the random seed.
            hyperparameters: Optional `HyperParameters` instance. Can be used to
                override (or register in advance) hyperparameters in the search
                space.
            tune_new_entries: Boolean, whether hyperparameter entries that are
                requested by the hypermodel but that were not specified in
                `hyperparameters` should be added to the search space, or not. If
                not, then the default value for these parameters will be used.
                Defaults to True.
            allow_new_entries: Boolean, whether the hypermodel is allowed to
                request hyperparameter entries not listed in `hyperparameters`.
                Defaults to True.
            max_retries_per_trial: Integer. Defaults to 0. The maximum number of
                times to retry a `Trial` if the trial crashed or the results are
                invalid.
            max_consecutive_failed_trials: Integer. Defaults to 3. The maximum
                number of consecutive failed `Trial`s. When this number is reached,
                the search will be stopped. A `Trial` is marked as failed when none
                of the retries succeeded.
        """
    def __init__(
        self,
        objective=None,
        initial_trials=10,
        estimator=None,
        max_trials=100,
        seed=None,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
    ):
        super().__init__(
            objective=objective,
            max_trials=max_trials,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            seed=seed,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        self.initial_trials = initial_trials
        self.estimator: Pipeline = estimator
        if self.estimator is None:
            raise ImplementationError("Estimator must be implemented")
        self.same_flag = False

    def populate_space(self, trial_id):
        # Still in initial search
        if int(trial_id)<self.initial_trials:
            values = self._random_values()
            if values is None:
                return {"status": TrialStatus.STOPPED, "values": None}
            return {"status": TrialStatus.RUNNING, "values": values}
        else:
            # Found a large enough sample, lets get to nudging
            X, y = self.build_estimator_training_data()
            feature, coeffs, signal = self.find_most_significant_error(X, y)
            best_params = self.get_best_trials(1)[0].hyperparameters
            next_params = self.nudge(best_params, feature, signal)
            return {"status": TrialStatus.RUNNING, "values": next_params}

    def nudge(self, params: HyperParameters, feature: int, sign):
        new_params = copy.deepcopy(params.values)
        key=list(params.values.keys())[feature]
        current_value = params.get(key)
        active_param = params.space[feature]
        if isinstance(active_param, Boolean):
            new_params[key] = not new_params[key]
        if isinstance(active_param, Float) or isinstance(active_param, Int):
            if sign == -1:
                new_params[key] = min(new_params[key]+active_param.step, active_param.max_value)
            elif sign == 1:
                new_params[key] = max(new_params[key]-active_param.step, active_param.min_value)
        if isinstance(active_param, Choice):
            index = active_param.values.index(current_value)
            if sign == -1:
                new_params[key] = active_param.values[(index + 1) % 4]
            elif sign == 1:
                new_params[key] = active_param.values[(index - 1) % 4]
        if new_params == params:
            print("[WARNING]Coefficients of Estimator are the same as last time, adding a random search to offset this")
            new_params = self._random_values()
        return new_params

    def find_most_significant_error(self, dataset, scores):
        self.estimator.fit(dataset, scores)
        coeffs = []
        for item in self.estimator.steps:
            if hasattr(item[1], "coef_"):
                coeffs.append(item[1].coef_)
        average_coeffs = np.array(coeffs).mean(axis=0)
        most_sig_feature = np.argmax(average_coeffs)
        return most_sig_feature, average_coeffs, np.sign(average_coeffs[most_sig_feature])

    def build_estimator_training_data(self):
        dataset = []
        scores = []
        for trial in self.trials.values():
            space: HyperParameters = trial.hyperparameters
            scores.append(trial.score)
            row = []
            for parameter in space.space:
                if isinstance(parameter, Boolean):
                    row.append(float(space.get(parameter.name)))
                if isinstance(parameter, Float) or isinstance(parameter, Int):
                    x = space.get(parameter.name)
                    mi = parameter.min_value
                    ma = parameter.max_value
                    x_scaled = (x - mi) / (ma - mi)
                    row.append(float(x_scaled))
                if isinstance(parameter, Choice):
                    row.append(float(list(map(lambda c: c == space.get(parameter.name), parameter.values)).index(True)))
            dataset.append(row)
        return np.array(dataset), np.array(scores)


class StatisticalSearch(Tuner):
    """Random search tuner.

    Args:
        hypermodel: Instance of `HyperModel` class (or callable that takes
            hyperparameters and returns a Model instance). It is optional when
            `Tuner.run_trial()` is overriden and does not use
            `self.hypermodel`.
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        max_trials: Integer, the total number of trials (model configurations)
            to test at most. Note that the oracle may interrupt the search
            before `max_trial` models have been tested if the search space has
            been exhausted. Defaults to 10.
        seed: Optional integer, the random seed.
        hyperparameters: Optional `HyperParameters` instance. Can be used to
            override (or register in advance) hyperparameters in the search
            space.
        tune_new_entries: Boolean, whether hyperparameter entries that are
            requested by the hypermodel but that were not specified in
            `hyperparameters` should be added to the search space, or not. If
            not, then the default value for these parameters will be used.
            Defaults to True.
        allow_new_entries: Boolean, whether the hypermodel is allowed to
            request hyperparameter entries not listed in `hyperparameters`.
            Defaults to True.
        max_retries_per_trial: Integer. Defaults to 0. The maximum number of
            times to retry a `Trial` if the trial crashed or the results are
            invalid.
        max_consecutive_failed_trials: Integer. Defaults to 3. The maximum
            number of consecutive failed `Trial`s. When this number is reached,
            the search will be stopped. A `Trial` is marked as failed when none
            of the retries succeeded.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses.
            Please see the docstring for `Tuner`.
    """

    def __init__(
        self,
        hypermodel=None,
        estimator=None,
        objective=None,
        initial_trials=10,
        max_trials=100,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs
    ):
        self.seed = seed
        oracle = StatisticalOracle(
            objective=objective,
            initial_trials=initial_trials,
            estimator=estimator,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super().__init__(oracle, hypermodel, **kwargs)







