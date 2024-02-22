import numpy as np
import torch
import copy
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch import nn

from .optrepo import OptRepo

class aggregator:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.opt = self._instantiate_opt()
    
    def _instantiate_opt(self):
        return OptRepo.name2cls(self.args.server_optimizer)(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.server_lr, 
            # momentum=self.args.server_momentum,
        )

    def set_model_global_grads(self, new_state):
        new_model = copy.deepcopy(self.model)
        new_model.load_state_dict(new_state)
        with torch.no_grad():
            for parameter, new_parameter in zip(
                    self.model.parameters(), new_model.parameters()
            ):
                parameter.grad = parameter.data - new_parameter.data
                # because we go to the opposite direction of the gradient
        model_state_dict = self.model.state_dict()
        new_model_state_dict = new_model.state_dict()
        for k in dict(self.model.named_parameters()).keys():
            new_model_state_dict[k] = model_state_dict[k]
        self.model.load_state_dict(new_model_state_dict)
    
    def fedavg(self, model_list):
        averaged_params = model_list[0]
        if self.args.client_num_per_round == 1:
            return averaged_params
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_model_params = model_list[i]
                # w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k]
                else:
                    averaged_params[k] += local_model_params[k]
        return averaged_params
    
    def fedopt(self, model, model_list):
        self.model = model
        averaged_params = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_model_params = model_list[i]
                # w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k]
                else:
                    averaged_params[k] += local_model_params[k]

        # server optimizer
        # save optimizer state
        self.opt.zero_grad()
        opt_state = self.opt.state_dict()
        # set new aggregated grad
        self.set_model_global_grads(averaged_params)
        self.opt = self._instantiate_opt()
        # load optimizer state
        self.opt.load_state_dict(opt_state)
        self.opt.step()
        return self.model.cpu().state_dict()
    
    def test_on_server(self, model, test_data, device):
        model.to(device)
        model.eval()
        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0, "test_f1": 0}
        criterion = nn.CrossEntropyLoss(reduction="sum").to(device)
        with torch.no_grad():
            label_list, pred_list = list(), list()
            for batch_idx, (data, labels) in enumerate(tqdm(test_data)):
                # for data, labels, lens in test_data:
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                loss = criterion(output, labels).data.item()
                pred = output.data.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability
                correct = pred.eq(labels.data.view_as(pred)).sum()
                for idx in range(len(labels)):
                    label_list.append(labels.detach().cpu().numpy()[idx])
                    pred_list.append(pred.detach().cpu().numpy()[idx][0])

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss * labels.size(0)
                metrics["test_total"] += labels.size(0)
        metrics["test_f1"] = f1_score(label_list, pred_list, average='macro')
        return metrics
            