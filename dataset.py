import torch.nn.functional as F
import torch
import numpy as np


'''----------------------------------------------data augmentation----------------------------------------------'''
def continus_mixup_data(*xs, y=None, alpha=0.5, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)
    # print(f'lam = {lam}')
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    y = lam * y + (1 - lam) * y[index]
    return *new_xs, y


'''----------------------------------------------ABIDE CC200----------------------------------------------'''
def load_data_abide_cc200(dataset_path, pick_site=None):
    """
        pcc_tensor: pcc
        one_hot_labels_tensor: one-hot
        ori_labels_tensor: label
        labels_np: 0,1
        site_np: site
    """
    data = np.load(dataset_path, allow_pickle=True).item()
    final_pearson = data["corr"]  # numpy
    labels_np = data["label"]  # numpy
    site_np = data['site']  # numpy

    # site to one-hot
    unique_sites = np.unique(site_np)
    num_classes = len(unique_sites)
    site_to_int = {site: idx for idx, site in enumerate(unique_sites)}
    site_int_encoded = np.array([site_to_int[site] for site in site_np])

    site_tensor_int = torch.tensor(site_int_encoded, dtype=torch.long)
    # one-hot
    site_one_hot = F.one_hot(site_tensor_int, num_classes=num_classes)

    # Numpy to Tensor
    pcc_tensor = torch.from_numpy(final_pearson).float()
    ori_labels_tensor = torch.from_numpy(labels_np).float()  #  0/1 label

    # one-Hot
    labels_long = torch.from_numpy(labels_np).long()  # long, onehot
    one_hot_labels_tensor = torch.nn.functional.one_hot(labels_long, num_classes=2)
    one_hot_labels_tensor = one_hot_labels_tensor.to(torch.float)  # float

    return pcc_tensor, one_hot_labels_tensor, ori_labels_tensor, labels_np, site_np, site_one_hot


'''----------------------------------------------ABIDE AAL116----------------------------------------------'''
def load_data_abide_aal116(dataset_path, pick_site=None):
    data = np.load(dataset_path, allow_pickle=True).item()
    pcc = data["pcc"]
    labels = data["labels"]
    site = data['sites']

    # numpy to tensor
    pcc, labels = [torch.from_numpy(data).float() for data in (pcc, labels)]

    ori_labels = labels

    # label to one-hot
    labels = labels.to(torch.long)  # long, onehot
    labels = torch.nn.functional.one_hot(labels, num_classes=2)
    one_hot_labels = labels.to(torch.float)  # float

    if pick_site is not None:
        nyu_mask = site == pick_site
        pcc = pcc[nyu_mask]
        one_hot_labels = one_hot_labels[nyu_mask]
        ori_labels = ori_labels[nyu_mask]

    return pcc, one_hot_labels, ori_labels


'''----------------------------------------------REST-meta-MDD CC200----------------------------------------------'''
def load_data_rest_meta_mdd(dataset_path):
    """
        pcc_tensor: pcc
        one_hot_labels_tensor: one-hot
        ori_labels_tensor: label
        labels_np: 0,1
        site_np: site
    """
    data = np.load(dataset_path, allow_pickle=True)

    r = data["cc200_pcc"]  # numpy
    labels_np = data["label"]  # numpy
    site_np = data["id"]  # numpy
    site_np = np.array([int(x.split('-')[0][1:]) for x in site_np])

    # x fisher z
    z = np.arctanh(r)
    N = z.shape[-1]
    z[:, np.arange(N), np.arange(N)] = 0
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    # Numpy to Tensor
    pcc_tensor = torch.from_numpy(z).float()
    one_hot_labels_tensor = torch.from_numpy(labels_np).float()

    return pcc_tensor, one_hot_labels_tensor, labels_np, site_np


'''----------------------------------------------BD MDD AAL116----------------------------------------------'''
def load_data_mdd_bd(dataset_path, label_path):
    data_pearson = np.load(dataset_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    # numpy to tensor
    data_pearson, labels = [torch.from_numpy(data).float() for data in (data_pearson, labels)]
    ori_labels = labels
    # label to one-hot
    labels = labels.to(torch.long)  # long, onehot
    labels = torch.nn.functional.one_hot(labels, num_classes=2)
    one_hot_labels = labels.to(torch.float)  # float

    return data_pearson, one_hot_labels, ori_labels











