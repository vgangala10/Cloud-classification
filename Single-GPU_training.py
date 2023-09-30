import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
import os
from tqdm import tqdm
from nn_t2v import *
import mlflow

os.chdir('/')

experiment_name = "cloud_training"
experiment_id = mlflow.create_experiment(experiment_name)

# Set the experiment for this run
mlflow.set_experiment(experiment_name)


torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TileNet = make_tilenet().to(device)
TileNet = TileNet.float()  # Convert the model to float type
TileNet = TileNet.to(device)

class TripletConcatDataset(Dataset):
    def __init__(self, concat_dataset):
        self.concat_dataset = concat_dataset
        self.dataset_lengths = [len(dataset) for dataset in concat_dataset.datasets]
        self.cumulative_lengths = [0] + list(np.cumsum(self.dataset_lengths))

    def __getitem__(self, index):
        # Find the corresponding dataset and index within that dataset
        for i, length in enumerate(self.cumulative_lengths[:-1]):
            if index >= length and index < self.cumulative_lengths[i + 1]:
                dataset_index = i
                dataset_specific_index = index - length

        # Access the corresponding dataset and retrieve the triplet
        specific_dataset = self.concat_dataset.datasets[dataset_index]
        patch, neighbor, distant = specific_dataset[dataset_specific_index]

        return patch, neighbor, distant

    def __len__(self):
        return sum(self.dataset_lengths)
# class triplet(Dataset):
#     def __init__(self, data):
#         self.patch = data[0]
#         self.neighbor = data[1]
#         self.distant = data[2]
#         self.length = data.shape[1]
#     def __getitem__(self, index):
#         return self.patch[index], self.neighbor[index], self.distant[index]
#     def __len__(self):
#         return self.length

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     init_process_group("nccl", rank=rank, world_size=world_size)

# from torch.utils.data.distributed import DistributedSampler
# def prepare(rank, world_size, data, batch_size=8, pin_memory=False, num_workers=0):
#     sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)  
#     dataloader = DataLoader(sampler, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
#     return dataloader


def transform_data(data_in):
    # applies some random flips and rotations to the data
    rand_i = np.random.rand()
    if rand_i < 0.2:
        # rotate 90
        data_in = data_in.transpose(2, 3).flip(2)
    elif rand_i < 0.4:
        # rotate 270
        data_in = data_in.transpose(2, 3).flip(3)
    elif rand_i < 0.6:
        # vert mirror
        data_in = data_in.flip(2)
    elif rand_i < 0.8:
        # horiz mirror
        data_in = data_in.flip(3)
    # else do nothing, use orig image

    return data_in
import time

start = time.time()

memmaps = [np.memmap('./storage/climate-memmap/triplet_data/orig_memmap'+str(i)+'.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128)) for i in range(2)]
print(memmaps[0].dtype)
data_ALL = ConcatDataset(memmaps)
data = TripletConcatDataset(data_ALL)
dataloader = DataLoader(data, batch_size=8, num_workers=2, drop_last=False, shuffle=False)

# memmaps = [np.memmap('./storage/climate-memmap/triplet_data/orig_memmap'+str(i)+'.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128)) for i in range(2)]
# mem_data = [i.transpose(1,0,2,3,4) for i in memmaps]
# data = np.concatenate(mem_data, axis = 0)
# data = triplet(data)
# dataloader = DataLoader(data, batch_size=8, num_workers=2, drop_last=False, shuffle=False)
lr = 0.001
optimizer = optim.Adam(TileNet.parameters(), lr=lr)
# model = DDP(TileNet, device_ids=[rank], output_device=rank, find_unused_parameters=True)
losses=[]
epochs = 2
# val_losses = []
# shape = data_val.shape
# patch_t, neighbor_t, distant_t = torch.from_numpy(data_val[0, :shape[1] ,...]), torch.from_numpy(data_val[1, :shape[1] ,...]), torch.from_numpy(data_val[2,:shape[1], ...])
# mlflow.start_run()
for epoch in range(0, epochs):
    running_loss=0
    TileNet.train()
    for i, data in tqdm(enumerate(dataloader,0)):
        # print(data[0].shape)
#             y = data.size()
#             data = data.permute(1, 0, 4, 2, 3).reshape(y[1], y[0], y[4], y[2], y[3])
        torch.cuda.empty_cache()
        a, n, d = data
        a, n, d = a.squeeze(), n.squeeze(), d.squeeze()
        if True:
            a = transform_data(a)
            n = transform_data(n)
            d = transform_data(d)
        a, n, d = (a.cuda(), n.cuda(), d.cuda())
        optimizer.zero_grad()
        loss, l_n, l_d, l_nd = TileNet.loss(a.float(), n.float(), d.float(), margin=1.0, l2= 0.0001)
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        a, n, d = a.cpu(), n.cpu(), d.cpu()
        loss, l_n, l_d, l_nd = loss.cpu(), l_n.cpu(), l_d.cpu(), l_nd.cpu()
        if i%20 == 0:
            print('\t  it', i,'     ep ',epoch,' loss', round(running_loss,3))
        mlflow.log_metric('running loss', round(running_loss,3), step = epoch * len(dataloader) + i)
    losses.append(running_loss)
    mlflow.log_metric('epoch loss', running_loss, step = epoch)
# destroy_process_group()
print(time.time()-start)
mlflow.end_run()
# 3022.35809469223 seconds for 20000 data points and 2 epochs with losses = [1200, 250]
