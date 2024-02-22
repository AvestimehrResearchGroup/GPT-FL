from mpi4py import MPI
import numpy as np
import sys
import os
import logging
import torch
import pdb
from tqdm import tqdm

class client:
    def __init__(self, args, rank_id, model_trainer):
        self.model = None
        self.args = args
        self.id = rank_id
        self.train_data_local_dict = None
        self.client_schedule = None
        self.trainer = model_trainer
        self.comm = None
        self.train_data_global = None

    def init_client(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        
        train_data_local_dict = comm.recv(source=0, tag=1)
        model = comm.recv(source=0, tag=2)
        client_schedule = comm.recv(source=0, tag=3)
        
        self.comm = comm
        self.model = model
        self.train_data_local_dict = train_data_local_dict
        self.client_schedule = client_schedule
        self.trainer.set_model(self, self.model, self.args)
        logging.info("rank %d has received initial messages" % (self.id))
    
    def client_agg(self, local_agg_params, model_weights, avg_weight=1.0):
        for name, param in model_weights.items():
            if name not in local_agg_params:
                local_agg_params[name] = param * avg_weight
            else:
                local_agg_params[name] += param * avg_weight
    
    def client_training(self):
        for round_idx in range(self.args.round):
            global_param = self.comm.recv(source = 0, tag = 4)
            logging.info("#######local training########### round_id = " + str(round_idx) + " at process " + str(self.id))
            local_agg_updates = {}
            for client_idx in self.client_schedule:
                logging.info("#######training with client index = %d ###########" % \
                    (client_idx))
                self.trainer.set_model_params(self, global_param)
                weights = self.trainer.train(self, self.train_data_local_dict[client_idx], self.args.device, round_idx, client_idx)
                self.client_agg(local_agg_updates, weights, avg_weight=1/self.args.client_num_per_round)
                logging.info("####### client index = %d finished local training ###########" % (client_idx))
            self.comm.send(local_agg_updates, dest = 0, tag = 1)
            self.client_schedule = self.comm.recv(source = 0, tag = 3)
    
            