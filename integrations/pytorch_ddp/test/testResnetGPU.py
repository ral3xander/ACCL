import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, barrier
#from mpi4py.MPI import COMM_WORLD as mpi
import torchvision.models as models

import argparse
import os
import sys
import time
import logging

# see https://rocm.blogs.amd.com/artificial-intelligence/ddp-training-pytorch/README.html

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get("ACCL_DEBUG") == "1" else logging.WARNING)


def train(model, loaders, optimizer, loss_fn, epochs, device, profiler=None):
    print(f"[Rank {rank}] Starting training on device {device}")
    start_time_train = time.perf_counter()
    model.train()
    total_step = len(loaders['train'])
    for epoch in range(epochs):
        
        start_time = time.perf_counter()
        for i, (images, labels) in enumerate(loaders['train']):
            images = images.to(device)
            labels = labels.to(device)
           
            if profiler: 
                profiler.step()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
        end_time = time.perf_counter()
        measured_time = (end_time - start_time)
        logger.debug ('rank: {} Epoch [{}/{}], Loss: {:.4f}, Time(s): {}' 
                .format(mpi.Get_rank(), epoch + 1, epochs, loss.item(), measured_time))
    end_time_train = time.perf_counter()
    measured_time_train = (end_time_train - start_time_train)
    print('Total train time: ' + str(measured_time_train))


def test(model, loaders, device):
    start_time_test = time.perf_counter()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time_test = time.perf_counter()
    measured_time_test = (end_time_test - start_time_test)
    print('Total test time: ' + str(measured_time_test))            
    print(f'Total accuracy: {correct}/{total} {correct/float(total)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',  choices=['resnet18', 'resnet34', 'resnet50'], default='resnet18')
    parser.add_argument('-a', '--master-address', type=str, default="localhost")
    parser.add_argument('-p', '--master-port', type=str, default="29500")
    
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    
    #global rank, size
    #rank = mpi.Get_rank()
    #size = mpi.Get_size()

    
    #device = torch.device(f"cuda:{rank}")
    init_process_group(backend="nccl")

    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"Hello from local_rank {local_rank}, global_rank {global_rank}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    
    torch.cuda.set_device(local_rank)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_sampler = DistributedSampler(trainset)

    loaders = {
        'train' : DataLoader(trainset, batch_size=128, shuffle=(train_sampler is None), sampler=train_sampler, pin_memory=True),
        'test'  : DataLoader(testset, batch_size=128, shuffle=False, pin_memory=True),
    }

    if args.model == 'resnet18':
        model = models.resnet18(num_classes=10)
    elif args.model == 'resnet34':
        model = models.resnet34(num_classes=10)
    elif args.model == 'resnet50':
        model = models.resnet50(num_classes=10)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove initial max pooling for CIFAR10 resolution

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
   
    epochs = 10
    batch_size = 128
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    profile = False
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2)

    if(profile):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./cifar_profiler_log'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            train(model, loaders, optimizer, loss_fn, epochs, local_rank, profiler=prof)
            test(model, loaders, local_rank)
    else:
            train(model, loaders, optimizer, loss_fn, epochs, local_rank)
            test(model, loaders, local_rank)

    barrier()
    destroy_process_group()