import torch.nn as nn
import torch
import numpy as np
from dpp_sampling import select_k_eigenvalues_from_L, select_samples_from_L, elementary_symmetric_polynomial
from sklearn.metrics.pairwise import pairwise_kernels



class Dropout():
    ''' Base class for dropout '''
    def __init__(self, dropout_proportion:float):
        self.dropout_proportion = dropout_proportion

    def update_kernel(self,data_loader):
        pass
    
    def apply_dropout(self):
        pass
        

class BernDrop(Dropout):
    def apply_dropout(self, x : torch.Tensor):
        ''' Keeps k values with equal probability '''

        k = int(x.shape[-1] * (1 - self.dropout_proportion))        
        mask = torch.ones_like(x)

        N = x.shape[-1]
        
        weights = torch.ones(N) 
        idxs = torch.multinomial(weights, N - k)
        mask = torch.zeros_like(x) 
        mask[:,idxs] = 1        
        x = x * mask

        return x, idxs                


class RBFDrop(Dropout):       
    def update_kernel(self,embeddings):                

        print('****** Updating DPP Kernel ******')                        
        embeddings = embeddings.transpose()
        N = embeddings.shape[-1]        
        self.k = int(embeddings.shape[0] * (1 - self.dropout_proportion))        

        # Step 1: Build the DPP
        with torch.no_grad():                        
            dist = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=-1)**2
            L = np.exp(- 10 * dist / N) 
            
            ##### Define the DPP 
            self.eig_vals, self.eig_vecs = np.linalg.eigh(L)


        self.elementary_symmetric_polynomials = elementary_symmetric_polynomial(self.k,self.eig_vals)
        
    
    def apply_dropout(self, x : torch.Tensor):
        ''' Keeps k values with equal probability '''
        device = x.device        

        B = x.shape[0] # batch size
        D = x.shape[-1] # dimension
        
        assert self.k <= D # can't have less samples than you are sampling                    
        
        # Step 2: Sample from the DPP once for each batch        
        vec_idx = select_k_eigenvalues_from_L(self.k,self.eig_vals, self.elementary_symmetric_polynomials) 
        sample_vecs = self.eig_vecs[:, vec_idx]        
        sample_idx = select_samples_from_L(sample_vecs)   

        idxs = torch.tensor(sample_idx)

        # Step 3: Build and apply our mask
        mask = torch.zeros_like(x) 
        mask[:,idxs] = 1        
          
        x = x * mask

        return x, idxs
        

class SineDrop(Dropout):       
    def update_kernel(self,embeddings):                

        print('****** Updating DPP Kernel ******')                        
        embeddings = embeddings.transpose()
        N = embeddings.shape[-1]        
        self.k = int(embeddings.shape[0] * (1 - self.dropout_proportion))        

        # Step 1: Build the DPP
        with torch.no_grad():                        
            norm_dist = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=-1)**2 / N
            L = np.sin(norm_dist * 5) / (norm_dist * 5)
            L[np.isnan(L)] = 1. # sin(x)/x = 1
            
            ##### Define the DPP 
            self.eig_vals, self.eig_vecs = np.linalg.eigh(L)


        self.elementary_symmetric_polynomials = elementary_symmetric_polynomial(self.k,self.eig_vals)
        
    
    def apply_dropout(self, x : torch.Tensor):
        ''' Keeps k values with equal probability '''
        device = x.device        

        B = x.shape[0] # batch size
        D = x.shape[-1] # dimension
        
        assert self.k <= D # can't have less samples than you are sampling                    
        
        # Step 2: Sample from the DPP once for each batch        
        vec_idx = select_k_eigenvalues_from_L(self.k,self.eig_vals, self.elementary_symmetric_polynomials) 
        sample_vecs = self.eig_vecs[:, vec_idx]        
        sample_idx = select_samples_from_L(sample_vecs)   

        idxs = torch.tensor(sample_idx)

        # Step 3: Build and apply our mask
        mask = torch.zeros_like(x) 
        mask[:,idxs] = 1        
          
        x = x * mask

        return x, idxs











# def dropout(features:torch.Tensor, x : torch.Tensor, dropout_proportion:float, dropout_type : 'str'):
#     # K is how many nodes we keep
#     k = int(features.shape[-1] * (1 - dropout_proportion))
    
#     if dropout_type == 'no_dropout':
#         return x
#     elif dropout_type == 'k_bernoulli':
#         return k_bernoulli_dropout(x,k)
#     elif dropout_type == 'rbf':
#         return rbf_dpp_dropout(features, x, k)
#     else:
#         raise ValueError(f'Dropout type: {dropout_type} is not defined')

# # def bernoulli_dropout(x : torch.Tensor, p : float):
# #     ''' Pytorch style dropout '''
# #     p_vec = (1-p) * torch.ones_like(x)    
# #     bern = torch.bernoulli(p_vec)    
# #     x = bern * x

# #     # Rescale to make testing mode easier
# #     x = x / (1-p)

# #     return x

# def k_bernoulli_dropout(x : torch.Tensor, k : int):
#     ''' Keeps k values with equal probability '''

#     mask = torch.ones_like(x)

#     N = x.shape[-1]
#     weights = torch.ones_like(x) / N # uniformly
#     idxs = torch.multinomial(weights, N - k)
            
#     mask.scatter_(1, idxs, 0.)

#     # Rescale to make testing mode easier
#     x = x * mask 
#     x = x / (k/N)    

#     return x    


# def rbf_dpp_dropout(features : torch.Tensor, x : torch.Tensor, k : int):              
#     device = x.device
#     N = features.shape[-1]
#     B = features.shape[0] # batch size
#     assert k <= N # can't have less samples than you are sampling    
    
#     # Step 1: Build the DPP
#     with torch.no_grad():
#         features = features.cpu().clone().numpy().transpose()
        
#         dist = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)**2
#         L = np.exp(- 20 * dist / N) 
        
#         ##### Define the DPP 
#         eig_vals, eig_vecs = np.linalg.eigh(L)

    

#     # Step 2: Sample from the DPP for each sample
#     idxs = torch.zeros(B,k, dtype = torch.int64, device = device)
#     for b in range(B):
#         vec_idx = select_k_eigenvalues_from_L(k,eig_vals) 
#         sample_vecs = eig_vecs[:, vec_idx]        
#         sample_idx = select_samples_from_L(sample_vecs)   

#         idxs[b,:] = torch.tensor(sample_idx)

#     # Step 3: Build and apply our mask
#     mask = torch.zeros_like(x) 
#     mask.scatter_(1,idxs,1.)

#     x = x / (1 - (N-k)/N)
#     x = x * mask

#     return x




# # def sine_dpp_dropout(features : torch.Tensor, x : torch.Tensor, k : int):        
# #     # Only keep the k-diverse samples
# #     # Proportion dropped out is (N-k) / N = 1 - p , p = k/N
    
# #     N = features.shape[-1]
# #     assert k <= N # can't have less samples than you are sampling

# #     mask = torch.zeros_like(x) # Will mask out dropped samples
    
    
# #     with torch.no_grad():
# #         features = features.cpu().clone().numpy()
# #         # Calculate distance between features
# #         dist = np.linalg.norm(features[:,None,:] - features[None,:,:], axis = -1) 

# #         # Calculate the sine kernel
# #         div_val = 0.01
# #         # L = np.sin(np.pi * dist / div_val) / (np.pi * dist / div_val) + 0.01 * np.eye(dist.shape[0])
# #         L = 0.01 * np.eye(dist.shape[0])
# #         L[np.isnan(L)] = 1
        
# #         ##### Get EigenValues
# #         eig_vals, eig_vecs = np.linalg.eigh(L)

# #         # Sample from k-DPP
# #         vec_idx = select_k_eigenvalues_from_L(k,eig_vals) 
# #         eig_vecs = eig_vecs[:, vec_idx]
        
# #         sample_idx = select_samples_from_L(eig_vecs)   
# #         mask[:,sample_idx] = 1

# #     # Rescale to make test mode easier 
# #     x = x / (1 - (N-k)/N)
# #     x = x * mask

# #     return x
