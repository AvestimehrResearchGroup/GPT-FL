from mpi4py import MPI
import numpy as np
import sys
import os
import copy
import logging
import wandb
import torch
import torchvision

# add the root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../model")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../data")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../synthetic_data_generation")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../synthetic_data_generation/model_check_point")))

from model.resnet import get_resnet
from model.pytorch_resnet import Resnet
from model.convnet import ConvNet
from torchvision.models import vgg19
import torch.nn as nn

from data.cifar10.dataset import CIFAR10_truncated
from data.cifar10.data_loader import load_partition_data_cifar10
from data.flower102.dataloader import load_partition_data_flower102
from data.cifar100.dataloader import load_partition_data_cifar100
from data.food101.dataloader import load_partition_data_food101
from data.eurosat.dataloader import load_partition_data_eurosat
from data.covidrad.dataloader import load_partition_data_covidrad

class server:
    def __init__(self, args, aggregator):
        self.model = None
        self.args = args
        self.data_package = None
        self.size = 0
        self.comm = None
        self.round_idx = 0
        self.aggregator = None

    def data_loading(self, dataset, partition_method, partition_alpha, client_number, batch_size):
        if dataset == "cifar10":
            logging.info('#########loading cifar10##########')
            data_dir = '../data/cifar10/cifar-10-batches-py'
            data_loader = load_partition_data_cifar10
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                data_dir,
                partition_method,
                partition_alpha,
                client_number,
                batch_size,
            )
        if dataset == "flower102":
            logging.info('#########loading flower102##########')
            data_loader = load_partition_data_flower102
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                partition_alpha,
                client_number,
                batch_size,
            )
        if dataset == "cifar100":
            logging.info('#########loading cifar100##########')
            data_loader = load_partition_data_cifar100
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                partition_alpha,
                client_number,
                batch_size,
            )
        if dataset == "food101":
            logging.info('#########loading food101##########')
            data_loader = load_partition_data_food101
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                partition_alpha,
                client_number,
                batch_size,
            )
        if dataset == "eurosat":
            logging.info('#########loading eurosat##########')
            data_loader = load_partition_data_eurosat
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                partition_alpha,
                client_number,
                batch_size,
            )
        if dataset == "covidrax":
           logging.info('#########loading covidrax##########')
           data_loader = load_partition_data_covidrad
           (
               train_data_num,
               test_data_num,
               train_data_global,
               test_data_global,
               train_data_local_num_dict,
               train_data_local_dict,
               test_data_local_dict,
               class_num,
           ) = data_loader(
               partition_alpha,
               client_number,
               batch_size,
           )

        
        return train_data_num, test_data_num, train_data_global, test_data_global, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

    def model_loading(self, model_name, output_dim):
        if model_name == 'resnet20':
            model = get_resnet(model_name)(output_dim, False)
        if model_name == 'resnet18':
            model = Resnet(
            num_classes=output_dim, resnet_size=18, pretrained=self.args.imagenet_pretrain_model)
        if model_name == 'resnet50':
            model = Resnet(
            num_classes=output_dim, resnet_size=50, pretrained=self.args.imagenet_pretrain_model)
        if model_name == 'vgg19':
            model = vgg19(pretrained = False)
            input_lastLayer = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(input_lastLayer,output_dim)
        if model_name == 'convnet':
            if self.args.dataset == 'flower102':
                model = ConvNet(num_classes=output_dim, im_size = (224,224))
            else:
                model = ConvNet(num_classes=output_dim, im_size = (32,32))
        self.model = model
        return model

    def client_sampling(self, client_per_round, client_in_total, size):
        sampling_list = np.random.choice(client_in_total, client_per_round)
        sampling_process = np.array_split(sampling_list, size)
        
        return sampling_process
    
    def init_server(self):
        # load data
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = self.data_loading(
            self.args.dataset,
            self.args.partition_method,
            self.args.partition_alpha,
            self.args.client_num_in_total,
            self.args.batch_size,
        )
        data_package = [train_data_num, test_data_global, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]

        self.data_package = data_package
        # load model
        model = self.model_loading(self.args.model, class_num)
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        self.size = size
        self.comm = comm
        
        sampling_process_list  = self.client_sampling(self.args.client_num_per_round, self.args.client_num_in_total, size-1)
        
        for i in range(1, self.size):
            comm.send(train_data_local_dict, dest=i, tag = 1)
            comm.send(model, dest=i, tag = 2)
            comm.send(sampling_process_list[i-1], dest = i, tag = 3)
            
        logging.info("Server has sent the data, model, and sampled number to each process")
    
    def server_training(self, aggregator):
        if self.args.self_pretrain_model:
            if self.args.model == 'resnet20':
                if self.args.dataset == 'cifar10':
                    logging.info('##########loading GPT-FL weight for cifar10+resnet20 ###########')
                    self.model.load_state_dict(torch.load("../synthetic_data_generation/model_check_point/resnet20_cifar10.pth"))

            if self.args.model == 'resnet18':
                if self.args.dataset == 'flower102':
                    if self.args.imagenet_pretrain_model == True:
                        logging.info('##########loading GPT-FL + imagenet weight for flower102+resnet18 ###########')
                        self.model.load_state_dict(torch.load("../synthetic_data_generation/model_check_point/resnet18_flower_3001.pth"))
                    else:
                        logging.info('##########loading GPT-FL weight for flower102+resnet18 ###########')
                        self.model.load_state_dict(torch.load("../synthetic_data_generation/model_check_point/resnet18_flower_3001.pth"))
 
                if self.args.dataset == 'food101':
                    if self.args.imagenet_pretrain_model == True:
                        logging.info('##########loading GPT-FL + imagenet weight for food101+resnet18 ###########')
                        self.model.load_state_dict(torch.load("../synthetic_data_generation/model_check_point/food101_syn_combined_resnet18_3001.pth"))
                    else:
                        logging.info('##########loading GPT-FL weight for food101+resnet18 ###########')
                        self.model.load_state_dict(torch.load("../synthetic_data_generation/model_check_point/food101_syn_combined_resnet18_0.0002_3001.pth"))

                if self.args.dataset == "cifar10":
                    if self.args.imagenet_pretrain_model == True:
                        logging.info('##########loading GPT-FL + imagenet weight for cifar10+resnet18 ###########')
                        self.model.load_state_dict(torch.load("../synthetic_data_generation/model_check_point/resnet18_cifar10_pre_1010.pth"))
        
            if self.args.model == 'resnet50':
                if self.args.dataset == 'cifar100':
                    if self.args.imagenet_pretrain_model == True:
                        logging.info('##########loading GPT-FL + imagenet weight for cifar100+resnet50 ###########')
                        self.model.load_state_dict(torch.load("../synthetic_data_generation/model_check_point/resnet50_cifar100_2006.pth"))
                    else:
                        logging.info('##########loading GPT-FL weight for cifar100+resnet50 ###########')
                        self.model.load_state_dict(torch.load("../synthetic_data_generation/model_check_point/resnet50_cifar100_pre_2007.pth"))

            if self.args.model == 'vgg19':
                if self.args.dataset == 'cifar10':
                    logging.info('##########loading GPT-FL weight for cifar10 + vgg19 ###########')
                    self.model.load_state_dict(torch.load("../synthetic_data_generation/model_check_point/vgg19_cifar10_4003.pth"))
                if self.args.dataset == 'cifar100':
                    logging.info('##########loading GPT-FL weight for cifar100 + vgg19 ###########')
                    self.model.load_state_dict(torch.load("../synthetic_data_generation/model_check_point/vgg19_cifar100_5001.pth"))

            if self.args.model == 'convnet':
                if self.args.dataset == 'cifar10':
                    logging.info('##########loading GPT-FL weight for cifar10 + convnet ###########')
                    self.model.load_state_dict(torch.load("../synthetic_data_generation/model_check_point/convnet_cifar10_1011.pth"))
                if self.args.dataset == 'cifar100':
                    logging.info('##########loading GPT-FL weight for cifar100 + convnet ###########')
                    self.model.load_state_dict(torch.load("../synthetic_data_generation/model_check_point/convnet_cifar100_2011.pth"))
                if self.args.dataset == 'flower102':
                    logging.info('##########loading GPT-FL weight for flower102 + convnet ###########')
                    self.model.load_state_dict(torch.load("../synthetic_data_generation/model_check_point/convnet_flower_3002.pth"))                   

        self.aggregator = aggregator(self.args, self.model)

        for round_idx in range(self.args.round):
            # give client the command to start training
            model_dic = {}
            for j in range(1, self.size):
                global_param = self.model.cpu().state_dict()
                self.comm.send(global_param, dest = j, tag = 4)
                # receive the local model updates
            for j in range(1, self.size):
                local_update = self.comm.recv(source = j, tag = 1)
                model_dic[j] = local_update
            if len(model_dic) != (self.size - 1):
                logging.info('server does not collect enough model updates')
                exit()
            else:
                model_list = []
                for idx in range(1, self.size):
                    model_list.append(model_dic[idx])
                logging.info('----------aggregating--------------')
                if self.args.aggregation == 'fedavg':
                    agg_global_params = self.aggregator.fedavg(model_list)
                if self.args.aggregation == 'fedopt':
                    agg_global_params = self.aggregator.fedopt(self.model, model_list)
                logging.info('----------aggregation finished--------------')
                self.model.load_state_dict(agg_global_params)
                
                if (round_idx % self.args.test_frequency == 0 or round_idx == self.args.round - 1):
                    logging.info("################testing on server : {}".format(round_idx))
                    self.server_testing(round_idx)
                    
                    
                sampling_process_list  = self.client_sampling(self.args.client_num_per_round, self.args.client_num_in_total, self.size-1)
                for j in range(1, self.size):
                    # self.comm.send(agg_global_params, dest = j, tag = 2)
                    self.comm.send(sampling_process_list[j-1], dest = j, tag = 3)
    
    def server_testing(self, round_idx):
        test_num_samples = []
        test_tot_corrects = []
        test_losses = []
        test_f1s = []
        test_performance = self.aggregator.test_on_server(self.model, self.data_package[1], self.args.device)
        test_tot_correct, test_num_sample, test_loss, test_f1 = (
            test_performance["test_correct"],
            test_performance["test_total"],
            test_performance["test_loss"],
            test_performance["test_f1"],
        )
        test_tot_corrects.append(copy.deepcopy(test_tot_correct))
        test_num_samples.append(copy.deepcopy(test_num_sample))
        test_losses.append(copy.deepcopy(test_loss))
        test_f1s.append(copy.deepcopy(test_f1))
        # test on test dataset
        test_acc = sum(test_tot_corrects) / sum(test_num_samples)
        test_loss = sum(test_losses) / sum(test_num_samples)
        wandb.log({"Test/Sample Number": sum(test_num_samples), "round": round_idx})
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/F1": test_f1, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        stats = {"test_acc": test_acc, 
                "test_f1": test_f1, 
                "test_loss": test_loss}
        logging.info(stats)

                