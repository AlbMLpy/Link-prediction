import torch
from torch.autograd import Variable, Function

class WARP(Function): 
    '''
    autograd function of WARP loss
    '''
    @staticmethod
    def forward(ctx, input, target, max_num_trials=None):
        
        batch_size = target.size()[0]
        if max_num_trials is None: 
            max_num_trials = target.size()[1] - 1
        
        positive_indices = torch.zeros(input.size(), device=device)
        negative_indices = torch.zeros(input.size(), device=device)
        L = torch.zeros(input.size()[0], device=device)
        
        all_labels_idx = np.arange(target.size()[1])
        
        Y = float(target.size()[1])
        J = torch.nonzero(target)
        
        for i in range(batch_size): 
            
            msk = np.ones(target.size()[1], dtype = bool)
            
            # For i user in batch choose ONE item!
            msk_J = J[:, 0] == i
            indice = np.random.choice(np.arange(J[msk_J].shape[0]), 1)
            #J[msk_J][indice].squeeze()[1]
            
            # Find the positive label for this example
            j = J[msk_J][indice].squeeze()[1]
            positive_indices[i, j] = 1
            msk[j] = False
            
            # initialize the sample_score_margin
            sample_score_margin = -1
            num_trials = 0
            
            neg_labels_idx = all_labels_idx[msk]

            while ((sample_score_margin < 0) and (num_trials < max_num_trials)):
                 
                #randomly sample a negative label
                neg_idx = np.random.choice(neg_labels_idx, 1)[0]
                msk[neg_idx] = False
                neg_labels_idx = all_labels_idx[msk]
                
                num_trials += 1
                # calculate the score margin 
                sample_score_margin = 1 + input[i, neg_idx] - input[i, j] 
            
            if sample_score_margin < 0:
                # checks if no violating examples have been found 
                continue
            else: 
                loss_weight = np.log(np.floor((Y-1)/(num_trials)))
                L[i] = loss_weight
                negative_indices[i, neg_idx] = 1
        
        loss = L * (
            1 - torch.sum(positive_indices*input, dim = 1) 
            + torch.sum(negative_indices*input, dim = 1)
        )
        
        ctx.save_for_backward(input, target)
        ctx.L = L
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices
        
        return torch.sum(loss , dim = 0, keepdim = True)

    # This function has only a single output, so it gets only one gradient 
    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        L = Variable(torch.unsqueeze(ctx.L, 1), requires_grad = False)

        positive_indices = Variable(ctx.positive_indices, requires_grad = False) 
        negative_indices = Variable(ctx.negative_indices, requires_grad = False)
        grad_input = grad_output * L * (negative_indices - positive_indices)

        return grad_input, None, None    

      
class WARPLoss(torch.nn.Module): 
    def __init__(self, max_num_trials = None): 
        super(WARPLoss, self).__init__()
        self.max_num_trials = max_num_trials
        
    def forward(self, input, target): 
        return WARP.apply(input, target, self.max_num_trials)