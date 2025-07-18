
from run import GSCon_run
import pickle
import argparse
import os
import torch

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='seed', default=53, type=int)

    argggs = parser.parse_args()
    return argggs


def get_args():
    argggs = parser_args()
    return argggs


def main():
    torch.set_num_threads(1)

    argggs = get_args()

    GSCon_run(model_name='GSCon', dataset_name='mosi', is_tune=False, seeds=[argggs.seed],argggs = argggs,model_save_dir="/data/sqhy_model/GSCon/MOSI/pt",
            res_save_dir="/data/sqhy_model/GSCon/MOSI/result", log_dir="/data/sqhy_model/GSCon/MOSI/log", mode='train')


if __name__ == '__main__':

    main()
