import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
    import accl_process_group as accl
    logger.debug("creating accl process group")
    if not simulator:
        rxbufsize = 4096 * 1024
        if host_file is None or fpga_file is None:
            sys.exit('Host and FPGA file need to be specified in hardware mode')
        with open(host_file, 'r') as hf, open(fpga_file, 'r') as ff:
            host_ips, fpga_ips = hf.read().splitlines(), ff.read().splitlines()
        ranks = [accl.Rank(ip, 5005 + i, i, rxbufsize) for i, ip in enumerate(fpga_ips)]
    else:
        rxbufsize = 4096 * 40
        ranks = [accl.Rank("127.0.0.1", 5500 + i, i, rxbufsize) for i in range(mpi.Get_size())]
    design = getattr(accl.ACCLDesign, comms)
    accl.create_process_group(ranks, design, bufsize=rxbufsize, initialize=True, simulation=simulator)


def train(model, loaders, optimizer, loss_fn, epochs, profiler=None):
    start_time_train = time.perf_counter()
    model.train()
    total_step = len(loaders['train'])
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            if (i-1) % 100 == 0 or i == 0: start_time = time.perf_counter()
            if profiler:
                profiler.step()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                end_time = time.perf_counter()
                measured_time = (end_time - start_time)
                logger.debug ('rank: {} Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time(s): {}' 
                       .format(mpi.Get_rank(), epoch + 1, epochs, i + 1, total_step, loss.item(), measured_time))
    end_time_train = time.perf_counter()
    measured_time_train = (end_time_train - start_time_train)
    print('Total train time: ' + str(measured_time_train))


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
    print('Total test time: ' + str(measured_time_test))            
    print(f'Total accuracy: {correct}/{total} {correct/float(total)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("-d", type=bool, default=None)
    parser.add_argument('-b', '--backend', choices=['accl', 'mpi'], default='mpi', help='Chose backend for collectives, either accl with fpga hardware support or software mpi')
    parser.add_argument('-m', '--model',  choices=['resnet18', 'resnet34', 'resnet50'], default='resnet18')
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

    if args.n == 1 and args.d is None:
        args.d = False
    elif args.n > 1 and args.d is None:
        args.d = True

    if args.d:
        if args.backend == 'mpi':
            torch.distributed.init_process_group("mpi", rank=rank, world_size=size)
        else:
            create_accl_process_group(args.simulator, args.comms, args.host_file, args.fpga_file)
            torch.distributed.init_process_group("ACCL", rank=rank, world_size=size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_sampler = DistributedSampler(trainset) if args.d else None

    loaders = {
        'train' : DataLoader(trainset, batch_size=100, shuffle=(train_sampler is None), sampler=train_sampler),
        'test'  : DataLoader(testset, batch_size=100, shuffle=False),
    }

    if args.model == 'resnet18':
        model = models.resnet18(num_classes=10)
    elif args.model == 'resnet34':
        model = models.resnet34(num_classes=10)
    elif args.model == 'resnet50':
        model = models.resnet50(num_classes=10)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove initial max pooling for CIFAR10 resolution

    if args.d:
        model = DDP(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./cifar_profiler_log'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        train(model, loaders, optimizer, loss_fn, epochs=10, profiler=prof)
        test(model, loaders)

    if args.d:
        torch.distributed.destroy_process_group()