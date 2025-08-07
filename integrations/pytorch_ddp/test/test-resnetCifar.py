import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    import accl_process_group as accl
    ACCL_AVAILABLE = True
except ImportError:
    ACCL_AVAILABLE = False
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, barrier
from mpi4py.MPI import COMM_WORLD as mpi
import torchvision.models as models

import argparse
import os
import sys
import time
import logging

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get("ACCL_DEBUG") == "1" else logging.WARNING)


def create_accl_process_group(simulator, comms, host_file, fpga_file):
    rxbufsize = 4194304
    #rxbufsize = 2097152
    logger.debug("creating accl process group")
    if not simulator:
        if host_file is None or fpga_file is None:
            sys.exit('Host and FPGA file need to be specified in hardware mode')
        with open(host_file, 'r') as hf, open(fpga_file, 'r') as ff:
            host_ips, fpga_ips = hf.read().splitlines(), ff.read().splitlines()
        ranks = [accl.Rank(ip, 5005, i, rxbufsize) for i, ip in enumerate(fpga_ips)]
    else:
        ranks = [accl.Rank("127.0.0.1", 5500 + i, i, rxbufsize) for i in range(mpi.Get_size())]


    if comms == 'udp':
        design = accl.ACCLDesign.udp
    elif comms == 'tcp':
        design = accl.ACCLDesign.tcp
    elif comms == 'cyt_rdma': # and not simulator:
        design = accl.ACCLDesign.cyt_rdma

    mpi.Barrier()
    logger.debug(f'Creating PG: \n Ranks: {ranks} \n Design: {design} \n Bufsize: {rxbufsize} \n Simulation: {simulator}')
    accl.create_process_group(ranks, design, bufsize=rxbufsize, initialize=True, simulation=simulator)


def train(model, loaders, optimizer, loss_fn, epochs, profiler=None):
    start_time_train = time.perf_counter()
    model.train()
    total_step = len(loaders['train'])
    for epoch in range(epochs):
        
        start_time = time.perf_counter()
        for i, (images, labels) in enumerate(loaders['train']):
        
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
    logger.debug('Total train time: ' + str(measured_time_train))


def test(model, loaders):
    start_time_test = time.perf_counter()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time_test = time.perf_counter()
    measured_time_test = (end_time_test - start_time_test)
    logger.debug('Total test time: ' + str(measured_time_test))            
    logger.debug(f'Total accuracy: {correct}/{total} {correct/float(total)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backend', choices=['accl', 'mpi'], default='mpi', help='Chose backend for collectives, either accl with fpga hardware support or software mpi')
    parser.add_argument('-m', '--model',  choices=['resnet18', 'resnet34', 'resnet50'], default='resnet50')
    parser.add_argument('-s','--simulator', action='store_true', default=False)
    parser.add_argument('-c', '--comms', choices=['udp', 'tcp', 'cyt_rdma', 'mpi'], default='cyt_rdma')
    parser.add_argument('-i', '--host-file', type=str)
    parser.add_argument('-f', '--fpga-file', type=str)
    parser.add_argument('-a', '--master-address', type=str, default="localhost")
    parser.add_argument('-p', '--master-port', type=str, default="29500")
    
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port

    rank = mpi.Get_rank()
    size = mpi.Get_size()

    if size > 1:
        if args.backend == 'mpi':
            init_process_group("mpi", rank=rank, world_size=size)
        else:
            create_accl_process_group(args.simulator, args.comms, args.host_file, args.fpga_file)
            init_process_group("ACCL", rank=rank, world_size=size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)



    reduced_fraction = 0.1
    subset_size_train = int(reduced_fraction * len(trainset))
    subset_size_test = int(reduced_fraction * len(testset))

    trainset = torch.utils.data.Subset(trainset, range(subset_size_train))
    testset = torch.utils.data.Subset(testset, range(subset_size_test))

    train_sampler = DistributedSampler(trainset) if size > 1 else None
    
    #, sampler=train_sampler
    loaders = {
        'train' : DataLoader(trainset, batch_size=128, shuffle=(train_sampler is None)),
        'test'  : DataLoader(testset, batch_size=128, shuffle=False),
    }

    if args.model == 'resnet18':
        model = models.resnet18(num_classes=10)
    elif args.model == 'resnet34':
        model = models.resnet34(num_classes=10)
    elif args.model == 'resnet50':
        model = models.resnet50(num_classes=10)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove initial max pooling for CIFAR10 resolution
    
    #bucket_cap_mb=1
    if size > 1 and args.backend == 'accl':
        model = DDP(model)
    elif size > 1 and args.backend == 'mpi':
         model = DDP(model)

    epochs = 13
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    
    profile = False
    schedule = torch.profiler.schedule(wait=40, warmup=0, active=80, repeat=1)

    if(profile):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./cifar_profiler_log'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            print()
            train(model, loaders, optimizer, loss_fn, epochs, profiler=prof)
            test(model, loaders)
    else:
            train(model, loaders, optimizer, loss_fn, epochs)
            test(model, loaders)

    if size > 1:
        accl.destroy()
        #destroy_process_group()