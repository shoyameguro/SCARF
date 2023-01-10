import logging

import hydra
import optuna
import torch

from train import Train
from utils import set_seed

logger = logging.getLogger(__name__)


class HyparaSearch:
    def __init__(self, config) -> None:
        self.seed = config['seed']
        self.direction = config['direction']
        self.n_trials = config['n_trials']
        self.train = Train(config)
        optuna.logging.get_logger("optuna").addHandler(logging.FileHandler('optuna.log'))

    def objective(self, trial):
        self.train.alpha = trial.suggest_categorical('self.train.alpha', [1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        self.train.beta = trial.suggest_categorical('self.train.beta', [1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        self.train.temperature = trial.suggest_categorical('self.train.temperature', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        self.train.c = trial.suggest_categorical('self.train.c', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.train.k = trial.suggest_int('self.train.k', 1, 4)

        set_seed(self.seed)
        self.train.self_ul()
        self.train.semi_sl()
        self.train.test()
        return 100 * torch.tensor(self.train.acc_results).mean()

    def get_best_params(self):
        study = optuna.create_study(direction=self.direction)
        study.optimize(self.objective, n_trials=self.n_trials)
        best_params = study.best_params
        return best_params


@hydra.main(config_path='conf', config_name='config')
def main(config):
    hyparasearch = HyparaSearch(config)
    best_params = hyparasearch.get_best_params()
    logger.info(f"[Best params]: {best_params}")


if __name__ == '__main__':
    main()
