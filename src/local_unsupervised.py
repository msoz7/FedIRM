from torch.utils.data import DataLoader, Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
from options import args_parser
from networks.convnext.convnextv2 import ConvNeXtV2, convnextv2_huge,convnextv2_base,convnextv2_large,convnextv2_atto,convnextv2_tiny,convnextv2_nano,convnext_pico
from networks.convnext.convnext import convnext_tiny
from utils import losses, ramps
from utils.util import get_timestamp
from confuse_matrix import get_confuse_matrix, kd_loss,update_pseudo_labels,get_confuse_covariance
import math
from torchvision import transforms
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.cluster import KMeans
import torch.nn as nn
from sklearn.decomposition import PCA


args = args_parser()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

class DatasetSplit(Dataset):
    
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
    
        items, index, strong_image_batch, weak_image_batch, label = self.dataset[self.idxs[item]]
        return items, index, strong_image_batch, weak_image_batch, label

class UnsupervisedLocalUpdate(object):
    
    def __init__(self, args, dataset=None, idxs=None):

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        self.epoch = 0
        self.iter_num = 0
        self.flag = True
        self.base_lr = 2e-4
        self.idxs_len=len(idxs)
    

    def train(self, args, net,ema_mod, itern,op_dict, epoch_param, target_matrix,target_matrix_covariance):
        net.train()

        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4
        )
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.base_lr

        self.epoch = epoch_param
        self.ema_model=ema_mod
        
        self.iter_num=itern
        epoch_loss = []
        epoch_accuracy = []
        used_pl = []
        print("begin training unsup")
        for epoch in range(args.local_ep):
            batch_loss = []
            accuracy = []
            iter_max = len(self.ldr_train)
            plcounter=0
            plmasked=0
         
            for i, (_,_,(strong_image_batch,ema_image_batch), weak_image_batch, label_batch) in enumerate(self.ldr_train):
                
                if torch.cuda.is_available():
                    strong_image_batch,weak_image_batch, ema_image_batch, label_batch = (
                        strong_image_batch.cuda(),
                        weak_image_batch.cuda(),
                        ema_image_batch.cuda(),
                        label_batch.cuda(),
                    )
                else:
                    image_batch, ema_image_batch, label_batch = (
                        image_batch,
                        ema_image_batch,
                        label_batch,
                    )

                ema_inputs = ema_image_batch
                # inputs = image_batch
                inputs = strong_image_batch

                outputs0,outputs = net(inputs)
                

                
                
                with torch.no_grad():
                    _,ema_output = self.ema_model(ema_inputs)
                T = 6
                with torch.no_grad():
                    # logits_sum,_ = net(inputs)
                    # for i in range(T):
                    #     logits,_ = net(inputs)
                    #     logits_sum = logits_sum + logits
                    # logits = logits_sum / (T + 1)

                    logits=outputs
                    preds = F.softmax(logits, dim=1)
                    uncertainty = -1.0 * torch.sum(
                        preds * torch.log(preds + 1e-6), dim=1
                    )
                    uncertainty_mask = uncertainty < 2.0

                with torch.no_grad():
                    activations = F.softmax(outputs, dim=1)
                    
                    confidence, _ = torch.max(activations, dim=1)
                    
                    sorted_numbers = sorted(confidence)
                    num_elements = len(confidence)
                    confidence_thresh=args.confidence_thresh
                    
                    confidence_mask = confidence >= confidence_thresh
                mask = confidence_mask * uncertainty_mask

                pseudo_labels = torch.argmax(activations[mask], dim=1)

                pseudo_labels = F.one_hot(pseudo_labels, num_classes=args.num_classes)
                


                #####################################################################################

                if((pseudo_labels.size()[0])==0):
                    with torch.no_grad():
                        confidence_thresh = (sorted_numbers[math.floor(num_elements*0.7)]).item()
                        confidence_mask = confidence >= confidence_thresh
                    mask = confidence_mask * uncertainty_mask
                    pseudo_labels = torch.argmax(activations[mask], dim=1)
                    pseudo_labels = F.one_hot(pseudo_labels, num_classes=args.num_classes)

                if args.network=="convnext":
                    source_matrix = get_confuse_matrix(outputs0[mask], pseudo_labels)
                    source_matrix_covariance=get_confuse_covariance(outputs[mask], pseudo_labels)
                else:
                    source_matrix = get_confuse_matrix(outputs[mask], pseudo_labels)
                    source_matrix_covariance=get_confuse_covariance(outputs[mask], pseudo_labels)

                plmasked=plmasked+(len(label_batch)-len(pseudo_labels))
                consistency_weight = get_current_consistency_weight(self.epoch)
                
     
                consistency_dist = (
                    torch.sum(losses.softmax_mse_loss(outputs, ema_output))
                    / (args.batch_size)
                )
                network_heat=15
                if args.network=="convnext":
                    network_heat=9

                consistency_loss = consistency_dist
                div_factor=1
                if args.confuse=="mean":
                    loss = (15 * consistency_weight * consistency_loss
                        + 
                        
                        (network_heat)*consistency_weight
                        *kd_loss(source_matrix, target_matrix)
                    )
                elif args.confuse=="covariance":
                    loss = (15 * consistency_weight * consistency_loss
                        + 
                        (network_heat)*consistency_weight
                        *kd_loss(source_matrix_covariance, target_matrix_covariance) 
                    )
                else:
                    loss = (15 * consistency_weight * consistency_loss
                        + 
                        (network_heat)*consistency_weight
                        *kd_loss(source_matrix, target_matrix)+ 
                        (network_heat)*consistency_weight
                        *kd_loss(source_matrix_covariance, target_matrix_covariance) 
                    )
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                update_ema_variables(net, self.ema_model, args.ema_decay, self.iter_num)
                batch_loss.append(loss.item())
                
                with torch.no_grad():
                    accuracy.append((torch.argmax(pseudo_labels, dim=1).cpu() == torch.argmax(label_batch[mask], dim=1).cpu()).float().mean())
                self.iter_num = self.iter_num + 1

            self.epoch = self.epoch + 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_accuracy.append((sum(accuracy) / len(accuracy)).item())
            used_pl.append((self.idxs_len-plmasked) /(self.idxs_len))
        print(f' loss:{epoch_loss}')
        print(f' accuracy:{epoch_accuracy}')
        print(f' plcounter:{plcounter} , plmasked:{plmasked+plcounter}')
            
        final_acc = (sum(epoch_accuracy) / len(epoch_accuracy))
        used_pl_ratio=(sum(used_pl) / len(used_pl))
 
        return (
            net.state_dict(),
            sum(epoch_loss) / len(epoch_loss),final_acc,copy.deepcopy(self.ema_model),itern,
            copy.deepcopy(self.optimizer.state_dict()),used_pl_ratio
        )
