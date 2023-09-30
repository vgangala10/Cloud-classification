import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
import os
from tqdm import tqdm
import mlflow
from nn_t2v import *

os.chdir('/')


## Dataloader class
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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)

from torch.utils.data.distributed import DistributedSampler
def prepare(rank, world_size, data, batch_size=8, pin_memory=True, num_workers=0):
    sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)  
    dataloader = DataLoader(data, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    return dataloader

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


from torch.nn.parallel import DistributedDataParallel as DDP
def main(rank, world_size):
    # setup the process groups
    setup(rank, world_size)
    # prepare the dataloader
    if rank == 0:
        experiment_name = "cloud_training"
        experiment_id = mlflow.create_experiment(experiment_name)

        # Set the experiment for this run
        mlflow.set_experiment(experiment_name)
    memmaps = [np.memmap('./storage/climate-memmap/triplet_data/orig_memmap'+str(i)+'.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128)) for i in range(2)]
    data_ALL = ConcatDataset(memmaps)
    data = TripletConcatDataset(data_ALL)
    batch_size = 8
    dataloader = prepare(rank, world_size, data, batch_size = batch_size)
    #Hello
    torch.cuda.set_device(rank)

    # instantiate the model(it's your own model) and move it to the right device
    TileNet = make_tilenet().float().to(rank)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    lr = 0.001
    model = DDP(TileNet, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses=[]
    epochs = 2
    # val_losses = []
    # shape = data_val.shape
    # patch_t, neighbor_t, distant_t = torch.from_numpy(data_val[0, :shape[1] ,...]), torch.from_numpy(data_val[1, :shape[1] ,...]), torch.from_numpy(data_val[2,:shape[1], ...])
    for epoch in range(0, epochs):
        running_loss=0
        TileNet.train()
        for i, data in tqdm(enumerate(dataloader,0)):
            # print(data[0].shape, len(data))
#             y = data.size()
#             data = data.permute(1, 0, 4, 2, 3).reshape(y[1], y[0], y[4], y[2], y[3])
            # torch.cuda.empty_cache()
            a, n, d = data
            a, n, d = a.squeeze(), n.squeeze(), d.squeeze()
            if True:
                a = transform_data(a) # augmenting the image
                n = transform_data(n)
                d = transform_data(d)
            a, n, d = (a.cuda(), n.cuda(), d.cuda())
            optimizer.zero_grad()
            loss, l_n, l_d, l_nd = model.module.loss(a.float(), n.float(), d.float(), margin=1.0, l2= 0.0001)
            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()
            # a, n, d = a.cpu(), n.cpu(), d.cpu()
            # loss, l_n, l_d, l_nd = loss.cpu(), l_n.cpu(), l_d.cpu(), l_nd.cpu()
            if rank == 0:
                print('\t  it', i,'     ep ',epoch,' loss', round(running_loss,3), ' step', len(dataloader), ' rank', rank)
                mlflow.log_metric('running_loss', round(running_loss,3), step = i)
        losses.append(running_loss)
        if rank == 0:
            mlflow.log_metric('epoch_loss', round(running_loss, 3), step = epoch)
    if rank == 0:
        mlflow.end_run()
    destroy_process_group()

    return
import time
start = time.time()
import sys
import torch.multiprocessing as mp
if __name__ == '__main__':
    # suppose we have 3 gpus
    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size
    )
# print(time.time()-start)
