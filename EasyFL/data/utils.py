import numpy as np

def direchlet_partition(
    file_label_list: list,
    num_subsets: int,
    alpha: float,
    seed: int=8,
    min_sample_size: int=5
) -> (list):
    
    # cut the data using dirichlet
    min_size = 0
    K, N = len(np.unique(file_label_list)), len(file_label_list)
    # seed
    np.random.seed(seed)
    while min_size < min_sample_size:
        file_idx_clients = [[] for _ in range(num_subsets)]
        for k in range(K):
            idx_k = np.where(np.array(file_label_list) == k)[0]
            np.random.shuffle(idx_k)
            # if self.args.dataset == "hateful_memes" and k == 0:
            #    proportions = np.random.dirichlet(np.repeat(1.0, self.args.num_clients))
            # else:
            proportions = np.random.dirichlet(np.repeat(alpha, num_subsets))
            # Balance
            proportions = np.array([p*(len(idx_j)<N/num_subsets) for p, idx_j in zip(proportions, file_idx_clients)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            file_idx_clients = [idx_j + idx.tolist() for idx_j,idx in zip(file_idx_clients,np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in file_idx_clients])
    return file_idx_clients