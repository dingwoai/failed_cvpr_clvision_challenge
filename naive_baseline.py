#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco, Massimo Caccia, Pau Rodriguez,        #
# Lorenzo Pellegrini. All rights reserved.                                     #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-02-2019                                                              #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""

Getting Started example for the CVPR 2020 CLVision Challenge. It will load the
data and create the submission file for you in the
cvpr_clvision_challenge/submissions directory.

"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os
import time
import copy
from core50.dataset import CORE50
import torch
import numpy as np
from utils.train_test import *
# import torchvision.models as models
from utils.resnet_mod import *
from utils.mobilenet_mod import *
from utils.mnasnet_mod import *
from utils.siamese import *
from utils.common import create_code_snapshot
from utils.radam import *


def main(args):

    # print args recap
    print(args, end="\n\n")

    # do not remove this line
    start = time.time()

    # Create the dataset object for example with the "ni, multi-task-nc, or nic
    # tracks" and assuming the core50 location in ./core50/data/
    dataset = CORE50(root='core50/data/', scenario=args.scenario,
                     preload=args.preload_data)

    # Get the validation set
    print("Recovering validation set...")
    full_valdidset = dataset.get_full_valid_set()

    # model
    if args.classifier == 'ResNet18':
        # classifier = models.resnet18(pretrained=True)
        classifier = resnet18(pretrained=True)
        classifier.fc = torch.nn.Linear(512, args.n_classes)
    
    if args.classifier == 'ResNet34':
        # classifier = models.resnet34(pretrained=True)
        classifier = resnet34(pretrained=True)
        classifier.fc = torch.nn.Linear(512, args.n_classes)

    if args.classifier == 'ResNet50':
        classifier = resnet50(pretrained=True)
        classifier.fc = torch.nn.Linear(2048, args.n_classes)

    if args.classifier == 'MobileNetV2':
        classifier = mobilenet_v2(pretrained=True)
        classifier.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1280, args.n_classes),
        )
    
    if args.classifier == 'mnasnet':
        classifier = mnasnet1_0(pretrained=True)
        classifier.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(1280, args.n_classes))
    
    if args.classifier == 'siamese':
        classifier = SiameseNetwork(n_class=args.n_classes)

    if args.optimizer =='sgd':
        opt = torch.optim.SGD(classifier.parameters(), lr=args.lr)
    if args.optimizer == 'adam':
        opt = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    if args.optimizer == 'radam':
        opt = RAdam(classifier.parameters(), lr=args.lr)
    # for param in classifier.conv1.parameters() or param in classifier.layer1.parameters() or param in classifier.relu.parameters():
    #     param.requires_grad = False
    # opt = torch.optim.SGD(filter(lambda p: p.requires_grad, classifier.parameters()), lr=args.lr)
    regularization_terms = {}
    task_count = 0
    criterion = torch.nn.CrossEntropyLoss()
    if args.classifier == 'siamese':
        # criterion = ContrastiveLoss(margin=2.)
        criterion = ContrastiveAndCELoss(margin=10.)

    # vars to update over time
    valid_acc = []
    ext_mem_sz = []
    ram_usage = []
    heads = []
    ext_mem = None

    # loop over the training incremental batches (x, y, t)
    for i, train_batch in enumerate(dataset):
        train_x, train_y, t = train_batch

        # adding eventual replay patterns to the current batch
        idxs_cur = np.random.choice(
            train_x.shape[0], args.replay_examples, replace=False
        )

        if i == 0:
            ext_mem = [train_x[idxs_cur], train_y[idxs_cur]]
        else:
            ext_mem = [
                np.concatenate((train_x[idxs_cur], ext_mem[0])),
                np.concatenate((train_y[idxs_cur], ext_mem[1]))]

        train_x = np.concatenate((train_x, ext_mem[0]))
        train_y = np.concatenate((train_y, ext_mem[1]))

        print("----------- batch {0} -------------".format(i))
        print("x shape: {0}, y shape: {1}, ext_mem_x shape: {2}, ext_mem_y shape: {3}"
              .format(train_x.shape, train_y.shape, ext_mem[0].shape, ext_mem[1].shape))
        print("Task Label: ", t)

        # train the classifier on the current batch/task
        # _, _, stats, regularization_terms, classifier = train_net_mrcl(
        #     opt, classifier, criterion, args.batch_size, train_x, train_y, i,
        #     regularization_terms, task_count, args.regularize_mode, args.icarl,
        #     args.epochs, preproc=preprocess_imgs
        # )
        i_es = 0
        lr = args.lr
        valid_acc2=[]
        for iepoch in range(args.epochs):
            if args.classifier != 'siamese':
                _, _, stats, regularization_terms = train_net(
                    opt, classifier, criterion, args.batch_size, train_x, train_y, i,
                    regularization_terms, task_count, args.regularize_mode, args.icarl, args.aug, args.resize_shape,
                    train_ep=1, preproc=preprocess_imgs
                )
            else:
                _, _, stats, regularization_terms = train_net_siamese(
                    opt, classifier, criterion, args.batch_size, train_x, train_y, i,
                    regularization_terms, task_count, args.regularize_mode, args.icarl, args.aug, args.resize_shape,
                    train_ep=1, preproc=preprocess_imgs
                )
            # stats2, _ = test_multitask(
            #     classifier, full_valdidset, args.batch_size, args.aug, args.resize_shape,
            #     preproc=preprocess_imgs, multi_heads=heads, verbose=False
            # )
            # valid_acc2 += stats2['acc']
            # print("{}th epoch, Avg. acc: {}".format(iepoch+1, stats2['acc']))
            # if stats2['acc']==max(valid_acc2):
            #     classifier_max = copy.deepcopy(classifier)
            #     print('update classifier using {}th'.format(iepoch+1))
            # else:
            #     i_es +=1  # counter for early stop
            #     if i_es>2:
            #         if args.optimizer =='sgd' and lr>=0.0001:
            #             lr = 0.1*lr
            #             opt = torch.optim.SGD(classifier.parameters(), lr=lr)
            #             i_es = 0
            #             print('learning rate reduced to {}'.format(lr))
            #         else:
            #             print('early stop at epoch {}'.format(iepoch+1))
            #             classifier = copy.deepcopy(classifier_max)
            #             break
        # classifier = copy.deepcopy(classifier_max)

        if args.scenario == "multi-task-nc":
            if args.classifier == 'mnasnet':
                heads.append(copy.deepcopy(classifier.classifier[1]))
            elif args.classifier == 'siamese':
                heads.append(copy.deepcopy(classifier.cnn1.classifier[1]))
            else:
                heads.append(copy.deepcopy(classifier.fc))

        ### not using the nearest neighbour classifier in icarl
        # if args.icarl:
        #     exem_class = []
        #     classifier = maybe_cuda(classifier)
        #     classifier.eval()
        #     # update exemplar features
        #     if args.classifier=='mnasnet':
        #         for i_class in range(args.n_classes):
        #             exemplar_features = []
        #             # nb_iters = (i_class==ext_mem[1]).sum()//args.batch_size + 1
        #             # x_i = torch.from_numpy(preprocess_imgs(ext_mem[0][i_class==ext_mem[1]])).type(torch.FloatTensor)

        #             nb_iters = (i_class==train_y).sum()//args.batch_size + 1
        #             x_i = torch.from_numpy(preprocess_imgs(train_x[i_class==train_y])).type(torch.FloatTensor)
        #             if len(x_i)==0:
        #                 print('no exemplars to be updated for class {}...'.format(i_class))
        #                 break
        #             with torch.no_grad():
        #                 for it in range(nb_iters):
        #                     start = it * args.batch_size
        #                     end = (it + 1) * args.batch_size
        #                     x_i_mb = maybe_cuda(x_i[start:end])
        #                     if x_i_mb.shape[0]==0:
        #                         break
        #                     feat_exem = classifier.layers(x_i_mb).mean([2, 3])  # mnasnet
        #                     exemplar_features.extend(np.array(feat_exem.cpu()))
        #                 mean_exem_feats = np.mean(exemplar_features, axis=0)
        #                 # mean_exem_feats = mean_exem_feats / np.linalg.norm(mean_exem_feats) # Normalize
        #                 exem_class.append(torch.from_numpy(mean_exem_feats).type(torch.FloatTensor))
        #         # classify with nearest 
        #         stats_icarl, _ = test_multitask_icarl(
        #             classifier, full_valdidset, exem_class, args.batch_size,
        #             preproc=preprocess_imgs, multi_heads=heads, verbose=False
        #         )
        #         print("icarl avg. acc: {}".format(stats_icarl['acc']))
        #     else:
        #         for i_class in range(args.n_classes):
        #             nb_iters = (i_class==ext_mem[1]).sum()//args.batch_size if (i_class==ext_mem[1]).sum()//args.batch_size >0 else (i_class==ext_mem[1]).sum()
        #             x_i = torch.from_numpy(preprocess_imgs(ext_mem[0][i_class==ext_mem[1]])).type(torch.FloatTensor)
        #             if len(x_i)==0:
        #                 break
        #             with torch.no_grad():
        #                 for it in range(nb_iters):
        #                     start = it * args.batch_size
        #                     end = (it + 1) * args.batch_size
        #                     x_i_mb = maybe_cuda(x_i[start:end])
        #                     feat_exem = classifier.conv1(x_i_mb)
        #                     feat_exem = classifier.bn1(feat_exem)
        #                     feat_exem = classifier.relu(feat_exem)
        #                     print(feat_exem.shape)
        #                     feat_exem = classifier.maxpool(feat_exem)
        #                     feat_exem = classifier.layer1(feat_exem)
        #                     feat_exem = classifier.layer2(feat_exem)
        #                     feat_exem = classifier.layer3(feat_exem)
        #                     feat_exem = classifier.layer4(feat_exem)
        #                     feat_exem = classifier.avgpool(feat_exem)
        #                     feat_exem = feat_exem.view(feat_exem.size(0), -1)
        #                     exemplar_features.append(np.array(feat_exem.cpu()))
        #                 mean_exem_feats = np.mean(np.mean(exemplar_features, axis=0), axis=0)
        #                 mean_exem_feats = mean_exem_feats / np.linalg.norm(mean_exem_feats) # Normalize
        #                 exem_class.append(torch.from_numpy(mean_exem_feats).type(torch.FloatTensor))
        #         # classify with nearest 
        #         stats_icarl, _ = test_multitask_icarl(
        #             classifier, full_valdidset, exem_class, args.batch_size,
        #             preproc=preprocess_imgs, multi_heads=heads, verbose=False
        #         )
        #         print("icarl avg. acc: {}".format(stats_icarl['acc']))
        ###

        task_count+=1

        # collect statistics
        ext_mem_sz += stats['disk']
        ram_usage += stats['ram']

        # test on the validation set
        if args.classifier != 'siamese':
            stats, _ = test_multitask(
                classifier, full_valdidset, args.batch_size, args.aug, args.resize_shape,
                preproc=preprocess_imgs, multi_heads=heads, verbose=False
            )
        else:
            exem_class = []
            classifier = maybe_cuda(classifier)
            classifier.eval()
            # update exemplar features
            for i_class in range(args.n_classes):
                exemplar_features = []
                nb_iters = (i_class==train_y).sum()//args.batch_size + 1
                x_i = preprocess_imgs(train_x[i_class==train_y], aug=False, resize_shape=args.resize_shape).type(torch.FloatTensor)
                if len(x_i)==0:
                    print('no exemplars to be updated for class {}...'.format(i_class))
                    break
                with torch.no_grad():
                    for it in range(nb_iters):
                        start = it * args.batch_size
                        end = (it + 1) * args.batch_size
                        x_i_mb = maybe_cuda(x_i[start:end])
                        if x_i_mb.shape[0]==0:
                            break
                        feat_exem = classifier.forward_once(x_i_mb)
                        i = random.choice(range(x_i_mb.shape[0]))  # sample one exemplar for each iteration per class
                        # exemplar_features.extend(np.array(feat_exem.cpu()))
                        exemplar_features.append(np.array(feat_exem[i].cpu()))
                    # mean_exem_feats = np.mean(exemplar_features, axis=0)
                    # # mean_exem_feats = mean_exem_feats / np.linalg.norm(mean_exem_feats) # Normalize
                    # exem_class.append(torch.from_numpy(mean_exem_feats).type(torch.FloatTensor))
                    exem_class.append(torch.from_numpy(np.array(exemplar_features)).type(torch.FloatTensor))
            # classify with nearest 
            stats, _ = test_multitask_siamese(
                classifier, full_valdidset, exem_class, args.batch_size, args.aug, args.resize_shape,
                preproc=preprocess_imgs, multi_heads=heads, verbose=False
            )

        valid_acc += stats['acc']
        print("------------------------------------------")
        print("Avg. acc: {}".format(stats['acc']))
        print("------------------------------------------")

    # Generate submission.zip
    # directory with the code snapshot to generate the results
    sub_dir = 'submissions/' + args.sub_dir
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # copy code
    create_code_snapshot(".", sub_dir + "/code_snapshot")

    # generating metadata.txt: with all the data used for the CLScore
    elapsed = (time.time() - start) / 60
    print("Training Time: {}m".format(elapsed))
    print("Final average valid acc: {}".format(np.average(valid_acc)))
    with open(sub_dir + "/metadata.txt", "w") as wf:
        for obj in [
            np.average(valid_acc), elapsed, np.average(ram_usage),
            np.max(ram_usage), np.average(ext_mem_sz), np.max(ext_mem_sz)
        ]:
            wf.write(str(obj) + "\n")

    # test_preds.txt: with a list of labels separated by "\n"
    print("Final inference on test set...")
    full_testset = dataset.get_full_test_set()
    if args.classifier != 'siamese':
        stats, preds = test_multitask(
            classifier, full_testset, args.batch_size, args.aug, args.resize_shape, preproc=preprocess_imgs,
            multi_heads=heads, verbose=False
        )
    else:
        stats, _ = test_multitask_siamese(
                classifier, full_testset, exem_class, args.batch_size, args.aug, args.resize_shape,
                preproc=preprocess_imgs, multi_heads=heads, verbose=False
            )

    with open(sub_dir + "/test_preds.txt", "w") as wf:
        for pred in preds:
            wf.write(str(pred) + "\n")

    print("Experiment completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # General
    parser.add_argument('--scenario', type=str, default="multi-task-nc",
                        choices=['ni', 'multi-task-nc', 'nic'])
    parser.add_argument('--preload_data', type=bool, default=True,
                        help='preload data into RAM')

    # Model
    parser.add_argument('-cls', '--classifier', type=str, default='ResNet18',
                        choices=['ResNet18', 'ResNet34', 'ResNet50', 'MobileNetV2', 'mnasnet', 'siamese'])

    # Optimization
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'radam'])
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--aug', type=bool, default=False,
                        help='whether to use train data augmentation')
    parser.add_argument('--resize_shape', type=int, default=128,
                        help='shape to be resized, eg. 128 224 256')

    # Continual Learning
    parser.add_argument('--regularize_mode', type=str, default=None,
                        choices=['L2', 'EWC', 'SI'])
    parser.add_argument('--icarl', type=bool, default=False)
    parser.add_argument('--replay_examples', type=int, default=0,
                        help='data examples to keep in memory for each batch '
                             'for replay.')

    # Misc
    parser.add_argument('--sub_dir', type=str, default="multi-task-nc",
                        help='directory of the submission file for this exp.')

    args = parser.parse_args()
    args.n_classes = 50
    args.input_size = [3, 128, 128]

    args.cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    main(args)
