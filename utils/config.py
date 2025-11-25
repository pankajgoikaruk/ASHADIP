import argparse, yaml, os


def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_args_with_config():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    return cfg


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)