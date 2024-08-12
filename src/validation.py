import os
import sys

import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd

import logging

import torch
from torch.nn import functional as F

from utils.metrics import compute_metrics_test


def epochVal_metrics_test(model, dataLoader, thresh):
    training = model.training
    model.eval()
    if torch.cuda.is_available():

        gt = torch.FloatTensor().cuda()
        pred = torch.FloatTensor().cuda()
    else:
        gt = torch.FloatTensor()
        pred = torch.FloatTensor()

    gt_study = {}
    pred_study = {}
    studies = []
    accuracy = []
    study_accuracy = []

    with torch.no_grad():
        for i, (study, _, image,_, label) in enumerate(dataLoader):
            if torch.cuda.is_available():

                image, label = image.cuda(), label.cuda()
                _,output = model(image)
            else:
                image, label = image, label
                _,output = model(image)
                #print(f"output{output}")
            output = F.softmax(output, dim=1)
            with torch.no_grad():
                accuracy.append((torch.argmax(output, dim=1).cpu() == torch.argmax(label, dim=1).cpu()).float().mean())
            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)
            with torch.no_grad():
                study_accuracy.append((torch.argmax(gt, dim=1).cpu() == torch.argmax(pred, dim=1).cpu()).float().mean())

        AUROCs, Accus, Senss, Specs, pre, F1,F1_weighted = compute_metrics_test(
            gt, pred, thresh=thresh, competition=True
        )

    model.train(training)
    #print(f"val accuracy: {sum(accuracy) / len(accuracy)}")
    #print(f"study val accuracy: {sum(accuracy) / len(accuracy)}")
    final_acc = (sum(accuracy) / len(accuracy))
    final_acc = final_acc.item()
    logging.info(
        "\n Test Validation Accuracy {:6f} Test Studies Validation Accuracy {:6f} ".format(
            sum(accuracy) / len(accuracy),sum(study_accuracy) / len(study_accuracy)
        )
    )


    return AUROCs,final_acc, Accus, Senss, Specs, pre, F1,F1_weighted
