import os,time,logging,random
import setproctitle,argparse
import numpy as np

import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

import Jdataset
from Jnetworks import JMCriterion
from Jnetworks.JMCriterion import all_reduce
from Jnetworksv2.JCMNetv8 import ParallelU_Net,NormalU_Net



local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
resume = ''
debug_flag = 0

if debug_flag == 1:
    from Jnetworks.JMCriterion import MCriterionData, MCriterion, Visualization

    mcdata = MCriterionData()
    mc_eval = MCriterion(include_background=True, datasets_flag="JCMNet_BraTS",
                         data=mcdata)


parser = argparse.ArgumentParser()
# Basic Inf
parser.add_argument('--user', default='JC', type=str)
parser.add_argument('--experiment', type=str, required=True)
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
parser.add_argument('--description',default= 'training logs-',type=str)

# DataSet Inf
parser.add_argument('--datasets_dir', type=str, required=True)
parser.add_argument('--train_file', type=str ,required=True)
parser.add_argument('--task', type=str, required=True)

# Device Inf
parser.add_argument('--gpus', type=str, required=True)
# Distribute training
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--DP_Method', default=0, type=int, help='Method Choose for DP Training, '
                                                             '0 for TorchDP,1 for Microsoft DP')
parser.add_argument('--amp', default=0, type=int, help='0 off, 1 use')

# Training Inf
parser.add_argument('--seed', default=2024, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--end_epoch', type=int, required=True)
parser.add_argument('--save_freq', default=1, type=int)

parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--num_workers', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--optimizer', type=str, required=True)

parser.add_argument('--weight_decay', default=1e-2, type=float)
parser.add_argument('--amsgrad', default=0, type=int)
parser.add_argument('--criterion', type=str, required=True)

# Eval on train Training
parser.add_argument('--valid_on_train', default=0, type=int)
parser.add_argument('--valid_file', type=str)
parser.add_argument('--valid_per_epoch', default=10, type=int)

args = parser.parse_args()

def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

def log_init():
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logs', args.experiment + args.date)
    log_file = log_dir + '.txt'
    log_args(log_file)
    logging.info('----------------------------------This is all args----------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('--------------------------------------------------------------------------------------------')
    logging.info('{}'.format(args.description))

def seed_fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def parameter_status(m):
    return str(sum([param.nelement() for param in m.parameters()]) / 1e6) + 'M'

def save_model(model,checkpoint_dir,epoch):

    if (epoch + 1) % int(args.save_freq) == 0 \
            or (epoch + 1) % int(args.end_epoch - 1) == 0 \
            or (epoch + 1) % int(args.end_epoch - 2) == 0 \
            or (epoch + 1) % int(args.end_epoch - 3) == 0:
        file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
        torch.save(model.state_dict(), file_name)

def print_model(model):
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        print(name, param)
        break

def choice_optimizer(model, args):
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay, momentum=0.99, nesterov=True)
    else:
        optimizer = None
    return optimizer

def maybe_update_lr(args, epoch, optimizer):
    optimizer.param_groups[0]['lr'] = poly_lr(epoch, args.end_epoch, args.lr, 0.9)

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d

def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data

def train(model, train_loader, criterion, optimizer, lr_sche, args, checkpoint_dir, epoch, rank):
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        if args.amp == 0:
            # default
            # train
            output = model(input)
            del input
            # loss
            loss, dice1, dice2, dice3 = criterion(output, target)
            # compute gradient and do optimizing step
            loss.backward()
            optimizer.step()
        else:
            amp_grad_scaler = GradScaler(enabled=True)
            with autocast():
                # train
                output = model(input)
                del input
                # loss
                loss, dice1, dice2, dice3 = criterion(output, target)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()

        reduce_loss = all_reduce(loss).detach().cpu().numpy()
        reduce_dice1 = all_reduce(dice1).detach().cpu().numpy()
        reduce_dice2 = all_reduce(dice2).detach().cpu().numpy()
        reduce_dice3 = all_reduce(dice3).detach().cpu().numpy()
        lr_sche.step()

        if rank == 0:
            logging.info('Epoch: {} Iteration:{}  loss: {:.5f} || dice_1:{:.4f} | dice_2:{:.4f} | dice_3:{:.4f} ||'
                         .format(epoch, i, reduce_loss, reduce_dice1, reduce_dice2, reduce_dice3))

    return model

def validate(model, valid_loader, criterion, optimizer, lr_sche, args, checkpoint_dir, epoch, rank):
    reduce_loss_sum = 0
    reduce_dice1_sum = 0
    reduce_dice2_sum = 0
    reduce_dice3_sum = 0
    if rank == 0:
        logging.info('===============================Validate===================================')
    for i, (input, target) in enumerate(valid_loader):
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()

            optimizer.zero_grad(set_to_none=True)
            output = model(input)

            loss, dice1, dice2, dice3 = criterion(output, target)

            reduce_loss = all_reduce(loss).detach().cpu().numpy()
            reduce_dice1 = all_reduce(dice1).detach().cpu().numpy()
            reduce_dice2 = all_reduce(dice2).detach().cpu().numpy()
            reduce_dice3 = all_reduce(dice3).detach().cpu().numpy()

            reduce_loss_sum += reduce_loss
            reduce_dice1_sum += reduce_dice1
            reduce_dice2_sum += reduce_dice2
            reduce_dice3_sum += reduce_dice3

            if rank == 0:
                logging.info('Validate: {} Iteration:{}  loss: {:.5f} || dice_1:{:.4f} | dice_2:{:.4f} | dice_3:{:.4f} ||'
                             .format(epoch, i, reduce_loss, reduce_dice1, reduce_dice2, reduce_dice3))
    if rank == 0:
        logging.info('==========================================================================')
        num = len(valid_loader)
        logging.info('Validate: {} Average : {:.5f} || dice_1:{:.4f} | dice_2:{:.4f} | dice_3:{:.4f} ||'
                     .format(epoch, reduce_loss_sum / num, reduce_dice1_sum / num, reduce_dice2_sum / num, reduce_dice3_sum / num))
        logging.info('=============================Validate end=================================')
    return model


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    log_init()
    seed_fix(args.seed)
    if args.DP_Method == 0:
        dist.init_process_group(backend="nccl")
    else:
        # deepspeed.init_distributed()
        pass
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    if args.amp == 0:
        logging.info("fix float train")
    else:
        logging.info("mix float train")

    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint',
                                  args.experiment + args.date)
    if rank == 0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)


    model = NormalU_Net(depths_unidirectional='small')

    logging.info(model)
    model.cuda()
    model = DDP(model)

    optimizer = choice_optimizer(model,args)
    criterion = JMCriterion.__dict__[args.criterion]().cuda()

    sche = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimizer, T_max = 50, eta_min = args.lr / 10, last_epoch = -1)

    if rank == 0:
        logging.info('train set file:' + args.train_file)
        if args.valid_on_train == 1:
            logging.info('valid set file:' + args.valid_file)
        logging.info("parameter:" + str(parameter_status(model)))


    train_root = os.path.join(args.datasets_dir)
    train_list = os.path.join(args.datasets_dir, args.train_file)
    train_datasets = Jdataset.__dict__[args.task](train_list, train_root, "train")
    if rank == 0:
        logging.info('Samples for train = {}'.format(len(train_datasets)))
    train_sampler = DistributedSampler(train_datasets)
    train_loader = DataLoader(dataset=train_datasets, sampler=train_sampler, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    valid_loader = None
    if args.valid_on_train == 1:
        valid_list = os.path.join(args.datasets_dir, args.valid_file)
        valid_datasets = Jdataset.__dict__[args.task](valid_list, train_root, "valid")
        if rank == 0:
            logging.info('Samples for valid = {}'.format(len(valid_datasets)))
        valid_sampler = DistributedSampler(valid_datasets)
        valid_loader = DataLoader(dataset=valid_datasets, sampler=valid_sampler, batch_size=args.batch_size,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.end_epoch):
        start_epoch = time.time()
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        if rank == 0:
            logging.info("lr:" + str(optimizer.param_groups[0]['lr']))

        model = train(model,train_loader,criterion,optimizer,sche,args,checkpoint_dir,epoch,rank)
        if args.valid_on_train == 1:
            if epoch % args.valid_per_epoch == 0 and epoch != 0:
                model = validate(model,valid_loader,criterion,optimizer,sche,args,checkpoint_dir,epoch,rank)

        if rank == 0:
            save_model(model,checkpoint_dir,epoch)

        end_epoch = time.time()

        if rank == 0:
            epoch_time_minute = (end_epoch-start_epoch)/60
            remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
            logging.info('Current epoch time: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining time: {:.2f} hours!'.format(remaining_time_hour))

    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))
    logging.info('----------------------------------The training process finished!-----------------------------------')
