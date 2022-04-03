import numpy as np

def corr_k(K,points):
    ''' The K-point correlation function given a determinantal kernel K and points'''
    Kp = K[points][:,points]

    if len(Kp.shape) > 1: 
        return np.linalg.det(Kp)
    else:
        return K

def select_eigenvalues_from_L(eig_vals):
    ''' Comprises Step 1 of the HKPV DPP sampling algorithm
    Select eigenvalues from the likelihood kernel L with eigenvalues = eig_vals
    
    Input: 
        eig_vals = Eigenvalues of likelihood kernel 
    Output:
        Indices of selected eigenvalues
    '''
    N = len(eig_vals)
    indices = np.arange(N)
    bern_probs = eig_vals / (eig_vals + 1)
    
    selected_indices = np.random.rand(N) <= bern_probs
            
    return indices[selected_indices]
    
def select_samples_from_L(V):
    ''' Comprises Step 2 of the HKPV DPP sampling algorithm and Step 2 of sampling from a k-DPP
    
    Input: 
        v = Eigenvectors corresponding to the selected eigenvalues obtained by 'select_eigenvalues_from_L'    
    Output: 
        Indices of selected samples coming from the DPP associated with the likelihood kernel L
    '''
    
    samples = [] 
    N, n_samples = V.shape
    indices = np.arange(N)    
    avail = np.ones(N, dtype=bool) 
    
    c = np.zeros((N, n_samples)) # Used for Gram-Schmidt
    normVT2 = (V**2).sum(axis = 1)
    
    for i in range(n_samples):
        probs = normVT2[avail] / (n_samples - i)  
        probs[probs < 0] = 0 # Compensate for small negatives due to round off        

        try:
            if np.abs(np.sum(probs) - 1) <= 0.05: # Allow 5% error
                probs = probs / np.sum(probs)

            j = np.random.choice(indices[avail], p=probs)
        except ValueError:                        
            raise ValueError(f'Probs don\'t sum to 1 but rather {np.sum(probs)}')


        samples.append(j)
        avail[j] = False
        # Cancel the contribution of V^T_j to the remaining feature vectors
        c[:, i] = (V @ V[j, :] - c[:, 0:i] @ c[j, 0:i]) / np.sqrt(normVT2[j])
        normVT2[:] -= c[:, i]**2

    return np.array(samples)


# K-DPP sampling requires computation of elementary symmetric polynomials
def elementary_symmetric_polynomial(k, eig_vals):
    ''' Compute the 1..kth elementary symmetric polynomials associated with the eigenvalues
    
    Input:
        k = the degree of the largest polynomial computed
        eig_vals = eigenvalues of likelihood kernel L
    Output: 
        The 1..kth elementary symmetric polynomials
     '''

    
    N = len(eig_vals)
    
    e = np.zeros([k + 1, N + 1])
    e[0, :] = 1.0

    for l in range(1, k + 1):
        for n in range(1, N + 1):
            e[l, n] = e[l, n - 1] + eig_vals[n - 1] * e[l - 1, n - 1]
    return e

def select_k_eigenvalues_from_L(k,eig_vals):
    ''' Step 1 in sampling from a k-DPP. Samples exactly k eigenvalues from the likelihood kernel L
    
    Input:
        k = number of eigenvalues to select
        eig_vals = eig_vals of the likelihood kernel L
    Output:
        Indices of selected eigenvalues
    '''

    e = elementary_symmetric_polynomial(k,eig_vals)
    # print(e)
    N = len(eig_vals)

    sample_idx = []
    for n in range(N,0,-1):
        thresh = eig_vals[n-1] * e[k-1,n-1] / e[k,n]        
        if np.isnan(thresh):
            raise ValueError('Sampling threshold is nan: Try sampling less points or using another DPP')
        if np.random.rand() <= thresh:
            sample_idx.append(n - 1) # subtract 1 to account for python's 0 indexing
            k -= 1
            if k == 0:
                break

    return sample_idx    
