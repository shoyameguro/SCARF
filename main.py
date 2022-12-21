import hydra

from train import Train
from utils import set_seed


@hydra.main(config_path='conf', config_name='config')
def main(config):
    set_seed(config['seed'])

    t = Train(config)
    if config['method'] == 'scarf':
        t.self_ul()
        # t.sl()

        t.semi_sl()
    t.test()


if __name__ == '__main__':
    main()
