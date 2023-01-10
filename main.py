import hydra

from train import Train
from utils import set_seed


@hydra.main(config_path='conf', config_name='config')
def main(config):
    set_seed(config['seed'])

    t = Train(config)
    if config['method'] == 'scarf':
        t.pretraining_scarf()
        t.sl()
    if config['method'] == 'original':
        t.self_ul()
        t.semi_sl()
    t.test()


if __name__ == '__main__':
    main()
