# Cloud-classififcation

Multi-GPU_training.py --> code for training 20000 triplets with 2 or more GPUs using pytorch DDP <br>
Single-GPU_training.py --> code for training 20000 triplets with one GPU<br>
nn_t2v.py --> ResNet-18 model code and loss metric<br>

A triplet consists of three 3x128x128 images. 
<ol>
  <li>Selected image</li>
  <li>Neighboring image</li>
  <li>Distant image</li>
</ol>

The dataset consists of 100 memmaps each of size 10000x3x3x128x128. 

The dataloader class is used to streamline all the memmaps and load them with the given batchsize.

The loss function measures the similarity and dissimilarity between Selected image and Neighboring image and, Selected image and the Distant image respectively.  
