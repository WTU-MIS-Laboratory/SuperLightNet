import argparse
import logging
import os
import sys
import time
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
import Jdataset
from Jnetworks.JMCriterion import MCriterionData, MCriterion
from Jnetworksv2.JCMNetv8 import NormalU_Net

GREEN = '\033[92m'
RED = '\033[91m'
END = '\033[0m'

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()
# Basic Inf
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
parser.add_argument('--mode', default='valid', type=str)

# DataSet Inf
parser.add_argument('--eval_dir',default=r'datasets\processed\MICCAI_BraTS_2019_Data_Training\Train', type=str)
parser.add_argument('--task', default='BraTS', type=str)

# Device Inf
parser.add_argument('--gpus', default='0', type=str)

# Eval Inf
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--output_dir', default='output', type=str)
parser.add_argument('--experiment', default='release_test', type=str)
parser.add_argument('--test_file', default='model_epoch_test.pth', type=str)
parser.add_argument('--eval_file', default=r'cross_validation\v3.txt', type=str)

args = parser.parse_args()


def init(args):
    load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'checkpoint', args.experiment, args.test_file)
    return load_file


def load_model(load_file, model):
    print(load_file)
    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint)
        model.eval()
        parameter_status(model)
    else:
        print('There is no resume file to load!')
        sys.exit(1)
    return model


def parameter_status(m):
    total = sum([param.nelement() for param in m.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))


def evaluate(model, eval_loader):
    mc_eval = MCriterion(include_background=True, datasets_flag="JCMNet_BraTS_New", data=MCriterionData())
    DSC, SDC = None, None
    for i, (input, target) in enumerate(eval_loader):
        print('-------------------------------' + str(i) + '------------------------------------')
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        DSC, SDC = mc_eval(output, target)
    return DSC, SDC


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # JCMNetv8
    model = NormalU_Net(depths_unidirectional='small')

    model.cuda()
    model = nn.DataParallel(model)

    print('eval set file:' + args.eval_file)
    eval_list = os.path.join(args.eval_dir, args.eval_file)
    eval_root = os.path.join(args.eval_dir)
    eval_datasets = Jdataset.__dict__[args.task](eval_list, eval_root, args.mode)
    print('Samples for eval = {}'.format(len(eval_datasets)))

    eval_loader = DataLoader(dataset=eval_datasets, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True, shuffle=False)

    load_file = init(args)
    model = load_model(load_file, model)
    with torch.no_grad():
        DSC, SDC = evaluate(model, eval_loader)
    logging.info(load_file)
    logging.info("DSC:" + str(DSC))
    logging.info("SDC:" + str(SDC))

