import torch
import random


def calculate_importance(params, regularization_terms, task_count, w, mode='EWC'):
    online_reg = True

    if mode=='L2':
        # Use an ident`ity importance so it is an L2 regularization.
        importance = {}
        for n, p in params.items():
            importance[n] = p.clone().detach().fill_(1)  # Identity

    elif mode=='EWC':
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.

        # Initialize the importance matrix
        if online_reg and len(regularization_terms)>0:
            importance = regularization_terms[0]['importance']
        else:
            importance = {}
            for n, p in params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized
        
        ## a modified version of EWC without per instance(batchsize=1) calculation, only gradient of last instance used. 
        ## time cost can be reduced.
        for n, p in importance.items():
            if params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                # p += ((self.params[n].grad ** 2) * len(input) / len(dataloader))
                p += (params[n].grad ** 2) 
    
    elif mode=='SI':
        online_reg = True
        damping_factor = 0.1
        # Initialize the importance matrix
        if len(regularization_terms)>0: # The case of after the first task
            importance = regularization_terms[0]['importance']
            prev_params = regularization_terms[0]['task_param']
        else:  # It is in the first task
            importance, prev_params = {}, {}
            for n, p in params.items():
                prev_params[n] = p.clone().detach()
                importance[n] = prev_params[n].fill_(0)  # zero initialized
            # prev_params = initial_params
        # Calculate or accumulate the Omega (the importance matrix)
        for n, p in importance.items():
            delta_theta = params[n].detach() - prev_params[n]
            p += w[n]/(delta_theta**2 + damping_factor)

    # Backup the weight of current task
    task_param = {}
    for n, p in params.items():
        task_param[n] = p.clone().detach()
    # Save the weight and importance of weights of current task
    if online_reg and len(regularization_terms)>0:
        # Always use only one slot in self.regularization_terms
        regularization_terms[0] = {'importance':importance, 'task_param':task_param}
    else:
        # Use a new slot to store the task-specific information
        regularization_terms[task_count] = {'importance':importance, 'task_param':task_param}
    print(task_count, 'reg_terms:', len(regularization_terms))
    # return importance, regularization_terms
    return regularization_terms


def l2_criterion(params, regularization_terms, reg_coef=3e2, regularization=True):
    loss = 0
    if regularization and len(regularization_terms)>0:
        # Calculate the reg_loss only when the regularization_terms exists
        reg_loss = 0
        for i,reg_term in regularization_terms.items():
            task_reg_loss = 0
            importance = reg_term['importance']
            task_param = reg_term['task_param']
            for n, p in params.items():
                task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
            reg_loss += task_reg_loss
        loss += reg_coef * reg_loss
    return loss

