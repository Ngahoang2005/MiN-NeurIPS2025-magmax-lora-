import json
import argparse
from trainer.BaseTrainer import train
import os
# Ép PyTorch không chia nhỏ bộ nhớ quá mức (giảm phân mảnh)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def main():
    args = setup_parser().parse_args()
    base_param = load_json(args.base_configs)
    model_param = load_json(args.model_configs)
    args = {**base_param, **model_param}
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Choose your configs.')
    parser.add_argument('--base_configs', type=str, default='./configs/Base_configs/',
                        help='Json file of base settings.')
    parser.add_argument('--model_configs', type=str, default='./configs/model_configs/',
                        help='Json file of model settings.')

    return parser


if __name__ == '__main__':
    main()
