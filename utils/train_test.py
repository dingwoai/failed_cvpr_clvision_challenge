#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco, Massimo Caccia, Pau Rodriguez,        #
# Lorenzo Pellegrini. All rights reserved.                                     #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 8-11-2019                                                              #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""
General useful functions for machine learning with Pytorch.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import torch
from torch.autograd import Variable
from .common import pad_data, shuffle_in_unison, check_ext_mem, check_ram_usage, shuffle_in_unison_siamese

import random
import copy
from .criterion_mod import *
from .augment import *
import torchvision.transforms.functional as F
from tqdm import tqdm

def train_net(optimizer, model, criterion, mb_size, x, y, t,
                regularization_terms, task_count, regularize_mode, icarl, aug, resize_shape,
              train_ep, preproc=None, use_cuda=True, mask=None):
    """
    Train a Pytorch model from pre-loaded tensors.

        Args:
            optimizer (object): the pytorch optimizer.
            model (object): the pytorch model to train.
            criterion (func): loss function.
            mb_size (int): mini-batch size.
            x (tensor): train data.
            y (tensor): train labels.
            t (int): task label.
            train_ep (int): number of training epochs.
            preproc (func): test iterations.
            use_cuda (bool): if we want to use gpu or cpu.
            mask (bool): if we want to maks out some classes from the results.
        Returns:
            ave_loss (float): average loss across the train set.
            acc (float): average accuracy over training.
            stats (dict): dictionary of several stats collected.
    """

    cur_ep = 0
    cur_train_t = t
    stats = {"ram": [], "disk": []}

    # if preproc:
    #     x = preproc(x)

    (train_x, train_y), it_x_ep = pad_data(
        [x, y], mb_size
    )

    shuffle_in_unison(
        [train_x, train_y], 0, in_place=True
    )

    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = None
    total_loss, l2_loss, distill_loss = 0, 0, 0

    if ~aug:
        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)
    
    if icarl:
        # Store network outputs with pre-update parameters
        model.eval()
        q = torch.zeros(len(train_x), 50).cuda()
        for it in range(it_x_ep):
            start = it * mb_size
            end = (it + 1) * mb_size
            optimizer.zero_grad()
            # x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)
            x_mb = preproc(train_x[start:end], aug, resize_shape)
            x_mb = maybe_cuda(x_mb, use_cuda=use_cuda)
            g = torch.nn.functional.sigmoid(model.forward(x_mb))
            q[start:end] = g.data
        q = Variable(q).cuda()

    for ep in range(train_ep):
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        w = {}
        for n,p in params.items():
            w[n] = p.clone().detach().zero_()
        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())

        model.active_perc_list = []
        model.train()

        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0
        for it in range(it_x_ep):

            start = it * mb_size
            end = (it + 1) * mb_size

            optimizer.zero_grad()

            x_mb = preproc(train_x[start:end], aug, resize_shape)
            x_mb = maybe_cuda(x_mb, use_cuda=use_cuda)
            # x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)
            y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)
            logits = model(x_mb)

            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == y_mb).sum()

            loss = criterion(logits, y_mb)
            if icarl:
                # calculate distillation loss
                q_i = q[start:end]
                g = torch.nn.functional.sigmoid(logits)
                distill_loss = torch.nn.BCELoss()(g, q_i)
                loss += distill_loss

            if regularize_mode is not None:
                # if len(regularization_terms)!=0:
                params = {n: p for n, p in model.named_parameters() if p.requires_grad}
                if regularize_mode in ['L2', 'EWC']:
                    l2_loss = l2_criterion(params, regularization_terms, reg_coef=1e3, regularization=True)
                    loss += l2_loss
                    loss.backward()
                    optimizer.step()
                elif regularize_mode=='SI':
                    unreg_gradients = {}
                    # 1.Save current parameters
                    old_params = {}
                    for n,p in params.items():
                        old_params[n] = p.clone().detach()
                    # 2. Collect the gradients without regularization term
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    for n, p in params.items():
                        if p.grad is not None:
                            unreg_gradients[n] = p.grad.clone().detach()
                    # 3. Normal update with regularization
                    l2_loss = l2_criterion(params, regularization_terms, reg_coef=5, regularization=True)
                    loss += l2_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # 4. Accumulate the w
                    for n, p in params.items():
                        delta = p.detach() - old_params[n]
                        if n in unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                            w[n] -= unreg_gradients[n] * delta  # w[n] is >=0
                # else:
                #     loss.backward()
                #     optimizer.step()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()

            acc = correct_cnt.item() / \
                  ((it + 1) * y_mb.size(0))
            ave_loss = total_loss/ ((it + 1) * y_mb.size(0))

            if it % 100 == 0:
                print(x_mb.min(), x_mb.max())
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )
                # print('loss', loss.item(), 'l2 loss:', l2_loss, 'distill loss:', distill_loss)
        cur_ep += 1

        if regularize_mode is not None:
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            regularization_terms = calculate_importance(params, regularization_terms, task_count, w, mode=regularize_mode)

    return ave_loss, acc, stats, regularization_terms


def preprocess_imgs(img_batch, aug, resize_shape, scale=True, norm=True, channel_first=True):
    """
    Here we get a batch of PIL imgs and we return them normalized as for
    the pytorch pre-trained models.

        Args:
            img_batch (tensor): batch of images.
            scale (bool): if we want to scale the images between 0 an 1.
            channel_first (bool): if the channel dimension is before of after
                                  the other dimensions (width and height).
            norm (bool): if we want to normalize them.
        Returns:
            tensor: pre-processed batch.

    """
    img_batch = batch_augment(img_batch, aug, resize_shape)

    if scale:
        # convert to float in [0, 1]
        img_batch = img_batch / 255

    if norm:
        # normalize
        img_batch[:, :, :, 0] = ((img_batch[:, :, :, 0] - 0.485) / 0.229)
        img_batch[:, :, :, 1] = ((img_batch[:, :, :, 1] - 0.456) / 0.224)
        img_batch[:, :, :, 2] = ((img_batch[:, :, :, 2] - 0.406) / 0.225)

    if channel_first:
        # Swap channel dimension to fit the caffe format (c, w, h)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))

    img_batch = torch.from_numpy(img_batch).type(torch.FloatTensor)
    return img_batch


def maybe_cuda(what, use_cuda=True, **kw):
    """
    Moves `what` to CUDA and returns it, if `use_cuda` and it's available.

        Args:
            what (object): any object to move to eventually gpu
            use_cuda (bool): if we want to use gpu or cpu.
        Returns
            object: the same object but eventually moved to gpu.
    """

    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda()
    return what


def test_multitask(
        model, test_set, mb_size, aug, resize_shape, preproc=None, use_cuda=True, multi_heads=[], verbose=True):
    """
    Test a model considering that the test set is composed of multiple tests
    one for each task.

        Args:
            model (nn.Module): the pytorch model to test.
            test_set (list): list of (x,y,t) test tuples.
            mb_size (int): mini-batch size.
            preproc (func): image preprocess function.
            use_cuda (bool): if we want to use gpu or cpu.
            multi_heads (list): ordered list of "heads" to be used for each
                                task.
        Returns:
            stats (float): collected stasts of the test including average and
                           per class accuracies.
    """

    model.eval()

    acc_x_task = []
    stats = {'accs': [], 'acc': []}
    preds = []

    for (x, y), t in test_set:

        # if preproc:
        #     x = preproc(x)

        if multi_heads != [] and len(multi_heads) > t:
            # we can use the stored head
            if verbose:
                print("Using head: ", t)
            with torch.no_grad():
                try:
                    model.fc.weight.copy_(multi_heads[t].weight)
                    model.fc.bias.copy_(multi_heads[t].bias)
                except:
                    model.classifier[1].weight.copy_(multi_heads[t].weight)
                    model.classifier[1].bias.copy_(multi_heads[t].bias)

        model = maybe_cuda(model, use_cuda=use_cuda)
        acc = None

        if ~aug:
            test_x = torch.from_numpy(x).type(torch.FloatTensor)
        test_y = torch.from_numpy(y).type(torch.LongTensor)

        correct_cnt, ave_loss = 0, 0

        with torch.no_grad():

            iters = test_y.size(0) // mb_size + 1
            for it in range(iters):

                start = it * mb_size
                end = (it + 1) * mb_size

                x_mb = preproc(test_x[start:end], aug, resize_shape)
                x_mb = maybe_cuda(x_mb, use_cuda=use_cuda)
                # x_mb = maybe_cuda(test_x[start:end], use_cuda=use_cuda)
                y_mb = maybe_cuda(test_y[start:end], use_cuda=use_cuda)
                logits = model(x_mb)

                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == y_mb).sum()
                preds += list(pred_label.data.cpu().numpy())

                # print(pred_label)
                # print(y_mb)
            print('nomal:',correct_cnt.item(),test_y.shape[0])
            acc = correct_cnt.item() / test_y.shape[0]

        if verbose:
            print('TEST Acc. Task {}==>>> acc: {:.3f}'.format(t, acc))
        acc_x_task.append(acc)
        stats['accs'].append(acc)

    stats['acc'].append(np.mean(acc_x_task))

    if verbose:
        print("------------------------------------------")
        print("Avg. acc:", stats['acc'])
        print("------------------------------------------")

    # reset the head for the next batch
    if multi_heads:
        if verbose:
            print("classifier reset...")
        with torch.no_grad():
            try:
                model.fc.weight.fill_(0)
                model.fc.bias.fill_(0)
            except:
                model.classifier[1].weight.fill_(0)
                model.classifier[1].bias.fill_(0)

    return stats, preds


def test_multitask_icarl(
        model, test_set, exem_class, mb_size, preproc=None, use_cuda=True, multi_heads=[], verbose=True):
    model.eval()

    acc_x_task = []
    stats = {'accs': [], 'acc': []}
    preds = []

    for (x, y), t in test_set:

        if preproc:
            x = preproc(x)

        model = maybe_cuda(model, use_cuda=use_cuda)
        acc = None

        test_x = torch.from_numpy(x).type(torch.FloatTensor)
        test_y = torch.from_numpy(y).type(torch.LongTensor)

        correct_cnt, ave_loss = 0, 0

        with torch.no_grad():

            iters = test_y.size(0) // mb_size + 1
            for it in range(iters):

                start = it * mb_size
                end = (it + 1) * mb_size

                x_mb = maybe_cuda(test_x[start:end], use_cuda=use_cuda)
                y_mb = maybe_cuda(test_y[start:end], use_cuda=use_cuda)
                feature = model.layers(x_mb).mean([2, 3]).cpu()

                means = torch.stack(exem_class) # (n_classes, feature_size)
                # means = torch.stack([means] * x_mb.shape[0]) # (batch_size, n_classes, feature_size)
                # means = means.transpose(1, 2) # (batch_size, feature_size, n_classes)

                # for i in range(feature.size(0)): # Normalize
                #     feature.data[i] = feature.data[i] / feature.data[i].norm()
                # feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
                # feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)

                # dists = (feature - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
                # if len(dists.shape)==1:
                #     dists = dists.unsqueeze(0)
                # _, pred_label = dists.min(1)

                ## cosine similarity
                pred_label = []
                for i in range(feature.size(0)): # Normalize
                    # feature.data[i] = feature.data[i] / feature.data[i].norm()
                    feat_temp = feature.data[i].expand_as(means)
                    dists = torch.cosine_similarity(means, feat_temp, dim=1)
                    p = dists.argmax()
                    pred_label.append(p)
                pred_label = torch.from_numpy(np.array(pred_label))
                correct_cnt += (pred_label == y_mb.cpu()).sum()
                preds += list(pred_label.data.cpu().numpy())

                # print(pred_label)
                # print(y_mb)
            print('icarl:',correct_cnt.item(),test_y.shape[0])
            acc = correct_cnt.item() / test_y.shape[0]

        if verbose:
            print('TEST Acc. Task {}==>>> acc: {:.3f}'.format(t, acc))
        acc_x_task.append(acc)
        stats['accs'].append(acc)

    stats['acc'].append(np.mean(acc_x_task))

    if verbose:
        print("------------------------------------------")
        print("Avg. acc:", stats['acc'])
        print("------------------------------------------")

    return stats, preds

def train_net_mrcl(optimizer, model, criterion, mb_size, x, y, t,
                regularization_terms, task_count, regularize_mode, icarl,
              train_ep, preproc=None, use_cuda=True, mask=None):
    cur_ep = 0
    cur_train_t = t
    stats = {"ram": [], "disk": []}

    if preproc:
        x = preproc(x)

    (train_x, train_y), it_x_ep = pad_data(
        [x, y], mb_size
    )

    shuffle_in_unison(
        [train_x, train_y], 0, in_place=True
    )

    model = maybe_cuda(model, use_cuda=use_cuda)
    model_inner = copy.deepcopy(model)
    model_outer = copy.deepcopy(model)
    optimizer_inner = torch.optim.Adam(model_inner.parameters(), lr=1e-4)
    optimizer_outer = torch.optim.Adam(model_outer.parameters(), lr=1e-4)
    acc = None
    total_loss, l2_loss = 0, 0

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    # train_ep = [3,3,3,3,2,2,2,2]
    for ep in range(train_ep):
        # params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        params_inner = {n: p for n, p in model_inner.named_parameters()}
        params_outer = {n: p for n, p in model_outer.named_parameters()}
        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())

        model_inner.active_perc_list = []
        model_outer.active_perc_list = []
        model_inner.train()
        model_outer.train()

        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0
        inner_cnt, outer_cnt, inner_steps, outer_steps = 0, 0, 5, 1
        # meta_losses = np.zeros(inner_steps)
        meta_loss = 0
        x_inner, y_inner = [], []
        
        for it in range(it_x_ep):

            start = it * mb_size
            end = (it + 1) * mb_size

            x_mb = batch_augment(train_x[start:end])

            x_mb = maybe_cuda(x_mb, use_cuda=use_cuda)
            y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)

            ## train PLN inner_steps times
            # update weights
            # use latest weights to calculate meta loss
            optimizer_inner.zero_grad()
            for n, p in params_inner.items():
                p.requires_grad = False
                if 'classifier' in n or 'fc' in n:
                    p.requires_grad = True
            logits = model_inner(x_mb)
            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == y_mb).sum()
            loss = criterion(logits, y_mb)
            loss.backward()
            optimizer_inner.step()
            
            # ## train RLN inner_steps times
            # if it%inner_steps==0 and it!=0:
            #     for outer_cnt in range(outer_steps):
            #         # copy latest weights of PLN
            #         # temp = copy.deepcopy(model_outer.classifier)
            #         model_outer.classifier = copy.deepcopy(model_inner.classifier)
            #         optimizer_outer = torch.optim.Adam(model_outer.parameters(), lr=1e-4)
            #         optimizer_outer.zero_grad()

            #         # randomly sample a batch of data to get meta loss
            #         it_rand = random.randint(1, it_x_ep-1)
            #         start = it_rand * mb_size
            #         end = (it_rand + 1) * mb_size
            #         x_rand = maybe_cuda(train_x[start:end], use_cuda=use_cuda)
            #         y_rand = maybe_cuda(train_y[start:end], use_cuda=use_cuda)
            #         # get meta loss
            #         logits = model_outer(x_rand)
            #         # model_outer.classifier = copy.deepcopy(temp)
            #         meta_loss = criterion(logits, y_rand)
            #         meta_loss.backward()
            #         optimizer_outer.step()
            #     model_inner = copy.deepcopy(model_outer)

            total_loss += loss.item()

            acc = correct_cnt.item() / \
                  ((it + 1) * y_mb.size(0))
            ave_loss = total_loss/ ((it + 1) * y_mb.size(0))

            if it % 100 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )
                print('loss', loss)
        cur_ep += 1
    model = copy.deepcopy(model_inner)
    return ave_loss, acc, stats, regularization_terms, model

def inner_update(self, x, fast_weights, y, bn_training):

    logits = self.net(x, fast_weights, bn_training=bn_training)
    loss = F.cross_entropy(logits, y)
    if fast_weights is None:
        fast_weights = self.net.parameters()
    #
    grad = torch.autograd.grad(loss, fast_weights)

    fast_weights = list(
        map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

    for params_old, params_new in zip(self.net.parameters(), fast_weights):
        params_new.learn = params_old.learn

    return fast_weights


def train_net_siamese(optimizer, model, criterion, mb_size, x, y, t,
                regularization_terms, task_count, regularize_mode, icarl, aug, resize_shape,
              train_ep, preproc=None, use_cuda=True, mask=None):
    cur_ep = 0
    cur_train_t = t
    stats = {"ram": [], "disk": []}

    # if preproc:
    #     x = preproc(x)

    (train_x, train_y), it_x_ep = pad_data(
        [x, y], mb_size
    )

    shuffle_in_unison(
        [train_x, train_y], 0, in_place=True
    )

    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = None
    total_loss, l2_loss, distill_loss = 0, 0, 0


    for ep in range(train_ep):
        ##shuffle to get different paired images for each epoch
        train_x0, train_x1, train_y0, train_y1, train_ys = shuffle_in_unison_siamese(
            train_x, train_y
        )
        if ~aug:
            train_x0 = torch.from_numpy(train_x0).type(torch.FloatTensor)
            train_x1 = torch.from_numpy(train_x1).type(torch.FloatTensor)
        train_y0 = torch.from_numpy(train_y0).type(torch.LongTensor)
        train_y1 = torch.from_numpy(train_y1).type(torch.LongTensor)
        train_ys = torch.from_numpy(train_ys).type(torch.LongTensor)

        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        w = {}
        for n,p in params.items():
            w[n] = p.clone().detach().zero_()
        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())

        model.active_perc_list = []
        model.train()

        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0
        for it in range(it_x_ep):

            start = it * mb_size
            end = (it + 1) * mb_size

            optimizer.zero_grad()

            x_mb_0 = preproc(train_x0[start:end], aug, resize_shape)
            x_mb_0 = maybe_cuda(x_mb_0, use_cuda=use_cuda)
            x_mb_1 = preproc(train_x1[start:end], aug, resize_shape)
            x_mb_1 = maybe_cuda(x_mb_1, use_cuda=use_cuda)
            y_mb_s = maybe_cuda(train_ys[start:end], use_cuda=use_cuda)
            y_mb_0 = maybe_cuda(train_y0[start:end], use_cuda=use_cuda)
            y_mb_1 = maybe_cuda(train_y1[start:end], use_cuda=use_cuda)

            output1, output2, output3, output4 = model(x_mb_0, x_mb_1)
            loss = criterion(output1, output2, output3, output4, y_mb_s, y_mb_0, y_mb_1)

            # euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
            # # cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
            # # print(euclidean_distance.max(), euclidean_distance.min())
            # pred_label = (euclidean_distance < 5)
            # # pred_label = (cosine_similarity > 0)
            # correct_cnt += (pred_label==y_mb).sum()

            _, pred_label = torch.max(output3, 1)
            correct_cnt += (pred_label == y_mb_0).sum()

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            acc = correct_cnt.item() / \
                  ((it + 1) * y_mb_0.size(0))
            ave_loss = total_loss/ ((it + 1) * y_mb_0.size(0))

            if it % 100 == 0:
                # print('pred_label sum thresh 1: ', pred_label.sum(), 'thresh 0.5:', (euclidean_distance < 0.5).sum(), 'thresh 0.8:', (euclidean_distance < .8).sum(), 'y_mb sum: ', y_mb_0.sum())
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )
                # print('loss', loss.item(), 'l2 loss:', l2_loss, 'distill loss:', distill_loss)
        cur_ep += 1

    return ave_loss, acc, stats, regularization_terms


def test_multitask_siamese(
        model, test_set, exem_class, mb_size, aug, resize_shape, preproc=None, use_cuda=True, multi_heads=[], verbose=True):

    model.eval()

    acc_x_task = []
    stats = {'accs': [], 'acc': []}
    preds = []

    for (x, y), t in test_set:

        # if preproc:
        #     x = preproc(x)

        if multi_heads != [] and len(multi_heads) > t:
            # we can use the stored head
            if verbose:
                print("Using head: ", t)
            with torch.no_grad():
                try:
                    model.fc.weight.copy_(multi_heads[t].weight)
                    model.fc.bias.copy_(multi_heads[t].bias)
                except:
                    try:
                        model.classifier[1].weight.copy_(multi_heads[t].weight)
                        model.classifier[1].bias.copy_(multi_heads[t].bias)
                    except:
                        model.cnn1.classifier[1].weight.copy_(multi_heads[t].weight)
                        model.cnn1.classifier[1].bias.copy_(multi_heads[t].bias)

        model = maybe_cuda(model, use_cuda=use_cuda)
        acc = None

        if ~aug:
            test_x = torch.from_numpy(x).type(torch.FloatTensor)
        test_y = torch.from_numpy(y).type(torch.LongTensor)

        correct_cnt, ave_loss = 0, 0

        with torch.no_grad():

            iters = test_y.size(0) // mb_size + 1
            for it in tqdm(range(iters)):

                start = it * mb_size
                end = (it + 1) * mb_size

                x_mb = preproc(test_x[start:end], aug, resize_shape)
                x_mb = maybe_cuda(x_mb, use_cuda=use_cuda)
                y_mb = maybe_cuda(test_y[start:end], use_cuda=use_cuda)

                feature = model.forward_once(x_mb).cpu()
                # means = torch.stack(exem_class)

                ## cosine similarity
                pred_label = []
                for i in range(feature.size(0)): # Normalize
                    temp = []
                    # feature.data[i] = feature.data[i] / feature.data[i].norm()
                    for exemplar in exem_class:
                        feat_temp = feature.data[i].expand_as(exemplar)
                        dists = torch.cosine_similarity(exemplar, feat_temp, dim=1)
                        # print('dists:', dists)
                        temp.append(dists.mean())
                    p = np.array(temp).argmax()
                    # feat_temp = feature.data[i].expand_as(means)
                    # dists = torch.cosine_similarity(means, feat_temp, dim=1)
                    # p = dists.argmax()
                    pred_label.append(p)
                pred_label = torch.from_numpy(np.array(pred_label))
                correct_cnt += (pred_label == y_mb.cpu()).sum()
                preds += list(pred_label.data.cpu().numpy())
                # logits = model.forward_orig(x_mb)
                # _, pred_label = torch.max(logits, 1)
                # correct_cnt += (pred_label == y_mb).sum()
                # preds += list(pred_label.data.cpu().numpy())
            print('nomal:',correct_cnt.item(),test_y.shape[0])
            acc = correct_cnt.item() / test_y.shape[0]

        if verbose:
            print('TEST Acc. Task {}==>>> acc: {:.3f}'.format(t, acc))
        acc_x_task.append(acc)
        stats['accs'].append(acc)

    stats['acc'].append(np.mean(acc_x_task))

    if verbose:
        print("------------------------------------------")
        print("Avg. acc:", stats['acc'])
        print("------------------------------------------")

    # reset the head for the next batch
    if multi_heads:
        if verbose:
            print("classifier reset...")
        with torch.no_grad():
            try:
                model.fc.weight.fill_(0)
                model.fc.bias.fill_(0)
            except:
                try:
                    model.classifier[1].weight.fill_(0)
                    model.classifier[1].bias.fill_(0)
                except:
                    model.cnn1.classifier[1].weight.fill_(0)
                    model.cnn1.classifier[1].bias.fill_(0)

    return stats, preds
