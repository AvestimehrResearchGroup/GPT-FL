import logging
import torch
from torch import nn
from tqdm import tqdm
import numpy as np

class cifar_res_trainer:
    def __init__(self):
        self.model = None
        self.args = None
        self.id = 0
        self.param_size = None
    
    def set_model(self, model, args):
        self.model = model
        self.args = args
        self.param_size = self.param_size = sum(p.numel() for p in self.model.parameters())
    
    def set_id(self, rank_id):
        self.id = rank_id
    
    def get_model_params(self):
        return self.model.cpu().state_dict()
    
    def set_model_params(self, model_params):
        self.model.load_state_dict(model_params)
        
    def train(self, train_data, device, round_idx, client_idx):
        model = self.model
        logging.info(" Client ID " + str(client_idx) + " round Idx " + str(round_idx))

        # count = 0
        # total_len = len(model.cpu().state_dict())
        # for p in model.parameters():
        #     count = count + 1
        #     if count >= total_len-1:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False

        model.to(device)
        model.train()  

        if self.args.precision == 'float16':
            model.half()  # convert to half precision
            for layer in model.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()
        if self.args.precision == 'float64':
            model.double()  # convert to double precision         

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
            # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            #  data, labels, lens
            for batch_idx, (data, labels) in enumerate(train_data):
                # data = torch.squeeze(data, 1)
                if self.args.precision == 'float16':
                    data = data.to(device).half()
                    labels = labels.to(device)
                elif self.args.precision == 'float64':
                    data = data.to(device).double()
                    labels = labels.to(device)
                else:
                    labels = labels.type(torch.LongTensor)
                    data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, labels)               
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_loss.append(loss.item())
                # break
            logging.info(
            "Client Index = {}\tEpoch: {}\tBatch Loss: {:.6f}\tBatch Number: {}".format(
                    client_idx, epoch, loss, batch_idx
                )
            )
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            
        local_update_state = model.cpu().state_dict()
        return local_update_state
            