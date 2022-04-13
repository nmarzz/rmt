import torch.nn as nn
import torch
import numpy as np
from dpp_sampling import select_k_eigenvalues_from_L, select_samples_from_L, elementary_symmetric_polynomial
from sklearn.metrics.pairwise import pairwise_kernels


class Dropout():
    ''' Base class for dropout '''

    def __init__(self, dropout_proportion: float):
        self.dropout_proportion = dropout_proportion

    def update_kernel(self):
        pass

    def apply_dropout(self):
        pass


class BernDrop(Dropout):
    def apply_dropout(self, x: torch.Tensor):
        ''' Keeps k values with equal probability '''

        k = int(x.shape[-1] * (1 - self.dropout_proportion))
        mask = torch.ones_like(x)

        N = x.shape[-1]

        weights = torch.ones(N)
        idxs = torch.multinomial(weights, N - k)
        mask = torch.zeros_like(x)
        mask[:, idxs] = 1
        x = x * mask

        return x, idxs


class RBFDrop(Dropout):
    def update_kernel(self, embeddings):

        print('****** Updating DPP Kernel ******')
        embeddings = embeddings.transpose()
        N = embeddings.shape[-1]
        self.k = int(embeddings.shape[0] * (1 - self.dropout_proportion))

        # Step 1: Build the DPP
        with torch.no_grad():
            dist = np.linalg.norm(
                embeddings[:, None, :] - embeddings[None, :, :], axis=-1)**2
            L = np.exp(- 10 * dist / N)

            # Define the DPP
            self.eig_vals, self.eig_vecs = np.linalg.eigh(L)

        self.elementary_symmetric_polynomials = elementary_symmetric_polynomial(
            self.k, self.eig_vals)

    def apply_dropout(self, x: torch.Tensor):
        ''' Keeps k values with equal probability '''
        device = x.device

        B = x.shape[0]  # batch size
        D = x.shape[-1]  # dimension

        assert self.k <= D  # can't have less samples than you are sampling

        # Step 2: Sample from the DPP once for each batch
        vec_idx = select_k_eigenvalues_from_L(
            self.k, self.eig_vals, self.elementary_symmetric_polynomials)
        sample_vecs = self.eig_vecs[:, vec_idx]
        sample_idx = select_samples_from_L(sample_vecs)

        idxs = torch.tensor(sample_idx)

        # Step 3: Build and apply our mask
        mask = torch.zeros_like(x)
        mask[:, idxs] = 1

        x = x * mask

        return x, idxs


class SineDrop(Dropout):
    def update_kernel(self, embeddings):

        print('****** Updating DPP Kernel ******')
        embeddings = embeddings.transpose()
        N = embeddings.shape[-1]
        self.k = int(embeddings.shape[0] * (1 - self.dropout_proportion))

        # Step 1: Build the DPP
        with torch.no_grad():
            norm_dist = np.linalg.norm(
                embeddings[:, None, :] - embeddings[None, :, :], axis=-1)**2 / N
            L = np.sin(norm_dist * 5) / (norm_dist * 5)
            L[np.isnan(L)] = 1.  # sin(x)/x = 1

            # Define the DPP
            self.eig_vals, self.eig_vecs = np.linalg.eigh(L)

        self.elementary_symmetric_polynomials = elementary_symmetric_polynomial(
            self.k, self.eig_vals)

    def apply_dropout(self, x: torch.Tensor):
        ''' Keeps k values with equal probability '''
        device = x.device

        B = x.shape[0]  # batch size
        D = x.shape[-1]  # dimension

        assert self.k <= D  # can't have less samples than you are sampling

        # Step 2: Sample from the DPP once for each batch
        vec_idx = select_k_eigenvalues_from_L(
            self.k, self.eig_vals, self.elementary_symmetric_polynomials)
        sample_vecs = self.eig_vecs[:, vec_idx]
        sample_idx = select_samples_from_L(sample_vecs)

        idxs = torch.tensor(sample_idx)

        # Step 3: Build and apply our mask
        mask = torch.zeros_like(x)
        mask[:, idxs] = 1

        x = x * mask

        return x, idxs
