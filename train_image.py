import os
import yaml
import argparse
import shutil

from models.meshnca import MeshNCA

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml', help="configuration")


def main(config):
    print(config['meshnca'])
    # model = MeshNCA(**config['meshnca'])

    print(config)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = config['experiment_name']
    exp_path = f'results/{exp_name}/'
    config['experiment_path'] = exp_path
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    shutil.copy(f'{args.config}', f'{exp_path}/config.yaml')
    main(config)
