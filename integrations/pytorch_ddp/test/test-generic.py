# /*****************************************************************************
#  Copyright (C) 2023 Advanced Micro Devices, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# *****************************************************************************/
from __future__ import annotations
from typing import Optional
import numpy as np
import os
import sys
import logging
import time
from mpi4py.MPI import COMM_WORLD as mpi

import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity
import accl_process_group as accl

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

#Configure logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logger = logging.getLogger(__name__)

if "ACCL_DEBUG" in os.environ and os.environ["ACCL_DEBUG"]=="1":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)
    
rank = 0
size = 0

x = 1024
y = 1



count = x * y
num_el = x * y
shape = (x , y)
#As in test.cpp defaults
rxbufsize = 4096 * 1024



def test_broadcast(numel, testtype):
    shape = (numel,)

    # testtype = torch.float32
    global num_errors

    if testtype == torch.int64 or testtype == torch.int32:
        rand_torch = torch.randint(torch.iinfo(testtype).min, torch.iinfo(testtype).max,shape, dtype=testtype)
        # rand_torch = torch.ones(shape, dtype=testtype)
    else:
        rand_torch = torch.rand(shape, dtype=testtype)
    
    # for i in range(10):
    if True:

        if rank == 0:
            x = rand_torch.clone()
        else:
            x = torch.ones(shape, dtype=testtype)

        mpi.Barrier()            
        
        with torch.profiler.record_function("test bcast "):

            start_time = time.perf_counter()

            dist.broadcast(x, 0)

            end_time = time.perf_counter()
            
        measured_time = (end_time - start_time) * 1000000

        print(str(rank) + "_pytorch_Broadcast_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
        
        mpi.Barrier()

        end_time = time.perf_counter()

        measured_time = (end_time - start_time) * 1000000


    try:
        np.testing.assert_allclose(x, rand_torch)
    except AssertionError as e:
        num_errors = num_errors + 1
        logger.debug("Test Broadcast failed")
        logger.debug(str(e))
    else:
        logger.debug("Test broadcast finished!")

def test_allreduce(numel, testtype):
    seed = 1234  # Choose a constant seed
    torch.manual_seed(seed)

    global num_errors

    shape = (numel,)

    
    if testtype == torch.int64 or testtype == torch.int32:
        rand_torch = torch.randint(torch.iinfo(testtype).min//size, torch.iinfo(testtype).max//size,shape, dtype=testtype)
    else:
        rand_torch = torch.rand(shape, dtype=testtype)
    
    # for i in range(10):
    if True:
    
        # shape = (320001,)
        x = rand_torch.clone()
     
        mpi.Barrier()            
        
        start_time = time.perf_counter()

        
        with torch.profiler.record_function("test_allreduce"):

            dist.all_reduce(x, dist.ReduceOp.SUM)

        end_time = time.perf_counter()
        measured_time = (end_time - start_time) * 1000000
        print(str(rank) + "_pytorch_Allreduce_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
        
        logger.debug("Directly measured time us 1:" + str(measured_time))            
        
        mpi.Barrier()
    
        try:
            np.testing.assert_allclose(x, rand_torch * size)
        except AssertionError as e:
            num_errors = num_errors + 1
            logger.debug("Test AllReduce failed")
            logger.debug(str(e))
        else:
            logger.debug("Test AllReduce finished!")

def test_reduce(numel):
    global num_errors


    shape = (numel,)
    x = torch.ones(shape)

    mpi.Barrier()            
    start_time = time.perf_counter()
    with torch.profiler.record_function("test_reduce"):

        dist.reduce(x, 0, dist.ReduceOp.SUM)
        mpi.Barrier()

    end_time = time.perf_counter()
    measured_time = (end_time - start_time) * 1000000
    print(str(rank) + "_pytorch_Reduce_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
    
    if rank == 0:
        try:
            np.testing.assert_allclose(x, torch.full(shape, float(size)))
        except AssertionError as e:
            num_errors = num_errors + 1
            logger.debug("Test Reduce failed")
            logger.debug(str(e))
        else:
            logger.debug("Test Reduce finished!")
  
def test_allgather(numel, testtype):
    global num_errors

    shape = (numel,)
    if testtype == torch.int64 or testtype == torch.int32:
        rand_torch = torch.randint(torch.iinfo(testtype).min, torch.iinfo(testtype).max,shape, dtype=testtype)
    else:
        rand_torch = torch.rand(shape, dtype=testtype)
    x = rand_torch.clone()
    y = [torch.full(shape, 0, dtype=testtype) for _ in range(size)]

    mpi.Barrier()            
    start_time = time.perf_counter()

    print('len y:' + str(len(y)))
    
    with torch.profiler.record_function("test_allgather"):
        dist.all_gather(y, x)

    end_time = time.perf_counter()
    measured_time = (end_time - start_time) * 1000000
    print(str(rank) + "_pytorch_Allgather_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
        
    mpi.Barrier()

        
    for i, c in enumerate(y):
        try:
            np.testing.assert_allclose(c, rand_torch)
        except AssertionError as e:
            num_errors = num_errors + 1
            logger.debug("Test AllGather failed")
            logger.debug(str(e))
        else:
            logger.debug("Test AllGather finished!")
        
def test_gather(numel):
    global num_errors

    shape = (numel,)
    x = torch.full(shape, float(rank))

    if rank == 0:
        y = [torch.empty(shape) for _ in range(size)]
    else:
        y = None

    mpi.Barrier()            
    start_time = time.perf_counter()
        
    with torch.profiler.record_function("test_gather"):
            
        dist.gather(x, y, 0)

    end_time = time.perf_counter()
    measured_time = (end_time - start_time) * 1000000
    print(str(rank) + "_pytorch_Gather_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
    
    if rank == 0:
        for i, c in enumerate(y):
            try:
                np.testing.assert_allclose(c, torch.full(shape, float(i)))
            except AssertionError as e:
                num_errors = num_errors + 1
                logger.debug("Test Gather failed")
                logger.debug(str(e))
            else:
                logger.debug("Test Gather finished!")

def test_scatter(numel):
    global num_errors

    shape = (numel,)
    if rank == 0:
        x = [torch.full(shape, float(i+1)) for i in range(size)]
    else:
        x = None
    y = torch.full(shape, float(0))

    mpi.Barrier()            
    start_time = time.perf_counter()
    
    with torch.profiler.record_function("test_scatter"):
        
        dist.scatter(y, x, 0)

    end_time = time.perf_counter()
    measured_time = (end_time - start_time) * 1000000
    print(str(rank) + "_pytorch_Scatter_" + str(y.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
    
    try:
        np.testing.assert_allclose(y, torch.full(shape, float(rank+1)))
    except AssertionError as e:
        num_errors = num_errors + 1
        logger.debug("Test Scatter failed")
        logger.debug(str(e))
    else:
        logger.debug("Test Scatter finished!")

def test_alltoall(numel):
    global num_errors

    # num_el = 26624
    
    shape = (numel,)

    input = torch.arange(numel, dtype=torch.float) + float(rank) * numel

    input_shaped = input.reshape(shape)

    output = torch.ones(numel)

    output_shaped = output.reshape(shape)

    start_time = time.perf_counter()
    
    with torch.profiler.record_function("test_alltoall"):
        
        dist.all_to_all_single(output_shaped, input_shaped)

    end_time = time.perf_counter()

    measured_time = (end_time - start_time) * 1000000
    
    print(str(rank) + "_pytorch_AlltoAll_" + str(input.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
        
    test = torch.zeros(numel)

    section_size = int(numel/size)

    for section in range(size):
        for el in range(section_size):
            test[section * section_size + el] = float(rank) * section_size + section * numel + el

    test_shaped = test.reshape(shape)
    try:
        np.testing.assert_allclose(output_shaped, test_shaped)
    except AssertionError as e:
        num_errors = num_errors + 1
        logger.debug("Test AlltoAll failed")
        logger.debug(str(e))
    else:
        logger.debug("Test AlltoAll finished!")            

def test_sendrcv(numel):
    global num_errors

    shape = (numel,)
    x = torch.full(shape, float(rank))

    y = torch.empty(shape)

    prev_rank = (rank - 1) % size
    next_rank = (rank + 1) % size


    with torch.profiler.record_function("test_sendrcv"):
        if rank % 2:
            mpi.Barrier()            
            start_time = time.perf_counter()
            dist.send(x, next_rank)
            end_time = time.perf_counter()
            measured_time = (end_time - start_time) * 1000000
            print(str(rank) + "_pytorch_Send_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)

            mpi.Barrier()            
            start_time = time.perf_counter()
            dist.recv(y, prev_rank)
            end_time = time.perf_counter()
            measured_time = (end_time - start_time) * 1000000
            print(str(rank) + "_pytorch_Recv_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
        else:
            mpi.Barrier()            
            start_time = time.perf_counter()
            dist.recv(y, prev_rank)
            end_time = time.perf_counter()
            measured_time = (end_time - start_time) * 1000000
            print(str(rank) + "_pytorch_Recv_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)

            mpi.Barrier()            
            start_time = time.perf_counter()
            dist.send(x, next_rank)
            end_time = time.perf_counter()
            measured_time = (end_time - start_time) * 1000000
            print(str(rank) + "_pytorch_Send_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
        mpi.Barrier()
    try:
        np.testing.assert_allclose(y, torch.full(shape, prev_rank))
    except AssertionError as e:
        num_errors = num_errors + 1
        logger.debug("Test Sendrcv failed")
        logger.debug(str(e))
    else:
        logger.debug("Test Sendrcv finished!")



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


def test_resNet18():
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        reduced_fraction = 0.01
        subset_size_train = int(reduced_fraction * len(trainset))
        subset_size_test = int(reduced_fraction * len(testset))

        trainset = torch.utils.data.Subset(trainset, range(subset_size_train))
        testset = torch.utils.data.Subset(testset, range(subset_size_test))

        train_sampler = DistributedSampler(trainset) if size > 1 else None

        loaders = {
            'train' : DataLoader(trainset, batch_size=128, shuffle=(train_sampler is None), sampler=train_sampler),
            'test'  : DataLoader(testset, batch_size=128, shuffle=False),
        }

        model = models.resnet18(num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # Remove initial max pooling for CIFAR10 resolution
        model = DDP(model, bucket_cap_mb=1)

        epochs = 1
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        train(model, loaders, optimizer, loss_fn, epochs)
        test(model, loaders)


def start_test(backend: str, comms: str, simulator: bool, host_file: str=None, fpga_file: str=None, ma: str="localhost", mp: str="30505"):
   
    global rank, size
    if ma==None:
        ma = "localhost"
    if mp==None:
        mp = "30505"
    os.environ['MASTER_ADDR'] = ma
    os.environ['MASTER_PORT'] = mp

    rank = mpi.Get_rank()
    size = mpi.Get_size()
    start_port = 5005
    logger.debug(f"Starting tests with the following parameters:\n\
Simulation: {simulator}, Communication Backend: {comms}\n\
Rank: {rank}, World size: {size}\n\
Host file: {host_file}, FPGA file: {fpga_file}\n\
Master address: {ma}:{mp}, Start port for FPGA: {start_port}")
    

    if not simulator:
        #default from test.cpp
        rxbufsize = 4096 * 1024
        if host_file==None or fpga_file==None: sys.exit('Host and FPGA file need to be specified in hardware mode')
            
        with open(host_file, 'r') as hf:
            host_ips = hf.read().splitlines()
            
        with open(fpga_file, 'r') as ff:
            fpga_ips = ff.read().splitlines()

        if comms == "cyt_rdma":
            ranks = [accl.Rank(a, start_port, i, rxbufsize) for i, a in enumerate(fpga_ips)]
        else:
            ranks = [accl.Rank(a, start_port + i, 0, rxbufsize) for i, a in enumerate(fpga_ips)]
    else:
        # Somehow the simulator gets stuck if I use the same rxbufsize
        rxbufsize = 4096
        ranks = [accl.Rank("127.0.0.1", 5500 + i, i, rxbufsize) for i in range(size)]

    logger.debug(f'Ranks: {ranks}')

    if comms == 'udp':
        design = accl.ACCLDesign.udp
    elif comms == 'tcp':
        design = accl.ACCLDesign.tcp
    elif comms == 'cyt_rdma': # and not simulator:
        design = accl.ACCLDesign.cyt_rdma
    # else:
        # if simulator:
            # sys.exit('Design "' + comms + '" currently not supported in simulator mode')
        # else:
            # sys.exit('Design "' + comms + '" currently not supported in hardware mode')

    # Sometimes ACCL gets stuck on the mpi import statement, so this is to avoid issues:
    mpi.Barrier()            


    # dist.init_process_group("mpi", rank=rank, world_size=size)
    if (simulator):
        logger.debug(f"Creating AcclPG with {ranks} ranks, {design} design,  simulation: {simulator}")
        accl.create_process_group(ranks, design, bufsize= rxbufsize , nbufs=16, initialize=True, simulation=simulator)
        logger.debug('Initialising accl backend')
        dist.init_process_group("ACCL", rank=rank, world_size=size)
    elif (backend == 'accl'):
        logger.debug(f"Creating AcclPG with {ranks} ranks, {design} design,  simulation: {simulator}")
        accl.create_process_group(ranks, design, bufsize= 4194304 , nbufs=16, initialize=True, simulation=simulator)
        logger.debug('Initialising accl backend')
        dist.init_process_group("ACCL", rank=rank, world_size=size)

    else:
        logger.debug('Initialising mpi backend')
        dist.init_process_group("mpi", rank=rank, world_size=size)

    
    global num_errors
    num_errors = 0

    if False:
        schedule = torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=1,
            repeat = 1,
        )
        

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                    profile_memory=True, schedule=schedule, record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler('./accl_log/profiler_log')) as prof:
            prof.step()
        # generic testing
        #test_resNet18()
    
    
    
    
    num = 32
    for n in range(0):
        test_allreduce(25524288, torch.float32)
        #test_allreduce(23654500, torch.float32)
        #test_allreduce(131072, torch.float32)
        #test_allreduce(64, torch.float32)
        #test_allreduce(2360, torch.float32)
        #test_allreduce(2360, torch.float32)
        #test_allreduce(2360, torch.float32)
        #test_broadcast(7000000, torch.float32)
        #test_broadcast(1, torch.float32)
        #test_broadcast(26, torch.float32)


    #Resnet18 sim

    #setup

    #batches
    for n in range(0):

        print()
        test_allreduce(2365450, torch.float32)
        test_allreduce(2360320, torch.float32)
        test_allreduce(2492416, torch.float32)
        test_allreduce(1180672, torch.float32)
        test_allreduce(590336, torch.float32)
        test_allreduce(590336, torch.float32)
        test_allreduce(623616, torch.float32)
        test_allreduce(295424, torch.float32)
        test_allreduce(295424, torch.float32)
        test_allreduce(267136, torch.float32)
        test_allreduce(112832, torch.float32)
    
    for n in range(0):

        print()
        #19
        test_allreduce(2490368, torch.float32)
        test_allreduce(2490368, torch.float32)
        #20
        test_allreduce(2621440, torch.float32)
        #10
        test_allreduce(1310720, torch.float32)
        #5
        test_allreduce(655360, torch.float32)
        test_allreduce(655360, torch.float32)
        test_allreduce(655360, torch.float32)
        #3
        test_allreduce(393216, torch.float32)
        test_allreduce(393216, torch.float32)
        test_allreduce(393216, torch.float32)
        #1
        test_allreduce(131072, torch.float32)
    
    for n in range(0):
        for i in range(93):
            test_allreduce(131072, torch.float32)
    
    for n in range(0):
        for i in range(93):
            test_allreduce(162144, torch.float32)


    for n in range(0):
        test_allreduce(num, torch.float32)
        test_reduce(num)
        test_allgather(num, torch.float32)
        test_gather(num)
        test_scatter(num)
        test_alltoall(num)
        test_sendrcv(num)
        
    for n in range(1):
        #test_allreduce(28928, torch.float32)
        #test_allreduce(524288, torch.float32)
        test_allreduce(112832, torch.float32)
        #test_allreduce(1050576, torch.float32)
        #test_alltoall(2024, torch.float32)

    
    if num_errors == 0:
        print("======== Successfully Finished testing======")
        logger.debug("======== Successfully Finished testing======")
    else:
        print(f"!!!!!!!! - {num_errors} Errors found - !!!!!!!!!")
        logger.debug(f"!!!!!!!! - {num_errors} Errors found - !!!!!!!!!")        

    # print(prof.key_averages(group_by_input_shape=True)
        # .table(sort_by="cpu_time_total", row_limit=15))
    schedule = torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=1,
            repeat = 1,
        )
        


    time.sleep(5)
    mpi.Barrier()
    logger.debug('Destroying ACCL Process Group')    
    accl.destroy()
    
    #dist.destroy_process_group()
       

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Coyote tests for ACCL ProcessGroup')
    parser.add_argument('-b', '--backend', choices=['accl', 'mpi'], default='mpi', help='Chose backend for collectives, either accl with fpga hardware support or software mpi')
    parser.add_argument('-s', '--simulation', action='store_true',
                        default=False, help='Use simulation instead of '
                                            'hardware')
    parser.add_argument('-c', '--comms', choices=['udp', 'tcp', 'cyt_rdma'], default='tcp',
                        help='Run tests over specified communication backend')
    parser.add_argument('-i', '--host-file', type=str, help='Specify the file, where the host IPs are listed')
    parser.add_argument('-f', '--fpga-file', type=str, help='Specify the file, where the FPGA IPs are listed')
    parser.add_argument('-a','--master-address', type=str)
    parser.add_argument('-p','--master-port', type=str)
    args = parser.parse_args()

    #if args.comms != 'cyt_rdma' or not args.simulation:
    #if args.comms != 'cyt_rdma':
    #    sys.exit('Currently only supports -c cyt_rdma and -s flags')
    start_test(args.backend, args.comms, args.simulation, args.host_file, args.fpga_file, args.master_address, args.master_port)
