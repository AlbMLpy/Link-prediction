import torch
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter

# Define model:
class GCP_WARP(torch.nn.Module):
    def __init__(self, n_entity, n_relation, n_factors=200, device='cpu'):
        super(GCP_WARP, self).__init__()
        
        # Objects and Subjects are represented as entity_factors:
        self.entity_factors = Parameter(
            torch.randn(n_entity, n_factors),
        )
        xavier_normal_(self.entity_factors)
        
        # Relations between the two entities:
        self.relations_factors = Parameter(
            torch.randn(n_relation, n_factors),
        )
        xavier_normal_(self.relations_factors)    
    
    def forward(self, subjects, relations, objects):
        # subjects, relations, objects - are indices
        
        pred = torch.sum(
            (self.entity_factors[subjects] 
            * self.relations_factors[relations]
            * self.entity_factors[objects]),
            dim=1,
        )
        return pred  
    
    def evaluate(self, data, hr_k=(3, 5, 10), how_many_samples=None):
        a = model.entity_factors.cpu().detach().numpy() #a = self.a_torch.cpu().data.numpy()
        b = model.relations_factors.cpu().detach().numpy() #b = self.b_torch.cpu().data.numpy()
        c = model.entity_factors.cpu().detach().numpy() #c = self.a_torch.cpu().data.numpy()
        
        a_norm = normalize(a, axis=1)
        b_norm = normalize(b, axis=1)
        c_norm = normalize(c, axis=1)
        
        print ("Count HR:", flush=True)

        if how_many_samples == None:
            triples = data.valid_triples
        else:
            triples = data.valid_triples[:how_many_samples]
        
        hr_result = ef.hr(
            data.filters,
            triples,
            a_norm, b_norm, c_norm,
            hr_k,
        )
        return hr_result