from mpi4py import MPI
import numpy as np
import sys
import os
import logging
import argparse
import torch
import random
import wandb

# add the root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../server")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../client")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from server.server_sim import server
from server.aggregation import aggregator
from client.client_sim import client
from client.local_trainer.cifar_resnet_trainer import cifar_res_trainer

def arg_setup(parser):
    # data related
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        help="cifar10, flower102, cifar100, food101, eurosat, covidrax",
    )
    
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.1, metavar='PA',
                        help='partition alpha (default: 0.1)')

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=100,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )

    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=10,
        metavar="NN",
        help="number of workers",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    # model related
    parser.add_argument(
        "--model",
        type=str,
        default="convnet",
        metavar="N",
        help="neural network used in training (resnet18, resnet20, resnet50, vgg19, convnet)",
    )
    # device related
    parser.add_argument(
        "--gpu_num_per_server", type=int, default=8, help="gpu_num_per_server"
    )

    parser.add_argument('--imagenet_pretrain_model', type=bool, default=False)

    parser.add_argument('--self_pretrain_model', type=bool, default=True, help='gpt-fl downstream models')

    parser.add_argument("--starting_gpu", type=int, default=0)
    
    parser.add_argument("--gpu_worker_num", type=int, default=8, help="total gpu num")
    # training related
    parser.add_argument("--round", type=int, default=1000, help="communication round")
    
    parser.add_argument("--epochs", type=int, default=1, help="local training epoch")
    
    parser.add_argument("--lr", type=float, default=0.1)
    
    parser.add_argument("--server_lr", type=float, default=1e-6)
    
    parser.add_argument("--aggregation", type=str, default='fedavg', help="aggregation function (fedavg, fedopt)")
    
    parser.add_argument("--test_frequency", type=int, default=5, help="testing frequency")
    
    parser.add_argument("--client_optimizer", type=str, default='sgd')

    parser.add_argument("--precision", type=str, default='float32')
    
    parser.add_argument("--server_optimizer", type=str, default='adam')
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    # set MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    process_id = comm.Get_rank()
    
    # set args
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logging.getLogger("PIL.TiffImagePlugin").setLevel(51)

    args = arg_setup(argparse.ArgumentParser(description="FL-Distributed"))
    logger.info(args)
    
    # set seeds
    set_seed(0)
    
    # set wandb
    if process_id == 0:
        wandb.init(
            mode = 'disabled',
            project = 'fediffusion',
            entity = 'ultraz',
            name = str(args.dataset) + "_FL_ran_init_" +str(args.aggregation),
            config = args,
        )
    
    # set devices
    if process_id == 0:
        device = torch.device(
            "cuda:" + str(args.starting_gpu) if torch.cuda.is_available() else "cpu"
        )
    else:
        process_gpu_dict = dict()
        for client_index in range(args.gpu_worker_num):
            gpu_index = client_index % args.gpu_num_per_server + args.starting_gpu
            process_gpu_dict[client_index] = gpu_index

        logging.info(process_gpu_dict)
        device = torch.device(
            "cuda:" + str(process_gpu_dict[process_id - 1])
            if torch.cuda.is_available()
            else "cpu"
        )
    args.device = device
    logger.info(device)
    
    # init server and client
    if process_id == 0:
        server_sim = server(args, aggregator)
        server_sim.init_server()
    else:
        model_trainer = cifar_res_trainer
        client_sim = client(args, process_id, model_trainer)
        client_sim.init_client()
    
    if process_id == 0:
        server_sim.server_training(aggregator)
    else:
        client_sim.client_training()
