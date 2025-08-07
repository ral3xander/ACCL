import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.distributed as dist
import accl_process_group as accl


from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from mpi4py.MPI import COMM_WORLD as mpi

from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

import argparse
import os
import sys
import logging
import time


def sync_hook(state, bucket):
    # Synchronize all ranks before all_reduce
    mpi.Barrier()   
    
    # Proceed with default all_reduce
    
    return allreduce_hook(state, bucket)

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logger = logging.getLogger(__name__)

if "ACCL_DEBUG" in os.environ and os.environ["ACCL_DEBUG"]=="1":
    logger.setLevel(logging.DEBUG)
else:
    #logger.setLevel(logging.WARNING)
    logger.setLevel(logging.DEBUG)

logger.debug(f"Rank: {mpi.Get_rank()} Python executable: {sys.executable}")

def create_accl_process_group(simulator, comms, host_file, fpga_file):
    logger.debug("creating accl process group")
    rxbufsize = 4194304
    if not simulator:
            #default from test.cpp
            
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
        
        ranks = [accl.Rank("127.0.0.1", 5500 + i, i, rxbufsize) for i in range(size)]
    

    if comms == 'udp':
        design = accl.ACCLDesign.udp
    elif comms == 'tcp':
        design = accl.ACCLDesign.tcp
    elif comms == 'cyt_rdma': # and not simulator:
        design = accl.ACCLDesign.cyt_rdma

    mpi.Barrier()
    logger.debug(f'Creating PG: \n Ranks: {ranks} \n Design: {design} \n Bufsize: {rxbufsize} \n Simulation: {simulator}')
    accl.create_process_group(ranks, design, bufsize=rxbufsize , nbufs=16, initialize=True, simulation=simulator)
    


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

def train(num_epochs, cnn, loaders, profiler=None):
    logger.debug("Start training")
    start_time_train = time.perf_counter()
    
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])

    optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   

    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        for i, (images, labels) in enumerate(loaders['train']):
            
            if profiler:
                p.step()

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]               

            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
        end_time = time.perf_counter()
        measured_time = (end_time - start_time)
        logger.debug ('rank: {} Epoch [{}/{}], Loss: {:.4f}, Time(s): {}'
                      .format(mpi.Get_rank(), epoch + 1, num_epochs, loss.item(), measured_time))

    end_time_train = time.perf_counter()
    measured_time_train = (end_time_train - start_time_train)

    logger.debug('Total train time: ' + str(measured_time_train))
        
def test():
    # Test the model
    start_time_test = time.perf_counter()
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct_current = (pred_y == labels).sum().item()
            total += labels.size(0)
            correct += correct_current
            
            print(f'Test Batch accuracy: {correct_current}/{labels.size(0)} {correct_current/float(labels.size(0))}')


    end_time_test = time.perf_counter()
    measured_time_test = (end_time_test - start_time_test)

    logger.debug('Total test time: ' + str(measured_time_test))            
    logger.debug(f'Total accuracy: {correct}/{total} {correct/float(total)}')
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backend', choices=['accl', 'mpi'], default='mpi', help='Chose backend for collectives, either accl with fpga hardware support or software mpi')
    parser.add_argument('-s', '--simulator', action='store_true', default=False, help='Use simulation instead of hardware')
    parser.add_argument('-c', '--comms', choices=['udp', 'tcp', 'cyt_rdma', 'mpi'], default='cyt_rdma', help='Run tests over specified communication backend')
    
    parser.add_argument('-i', '--host-file', type=str, help='Specify the file, where the host IPs are listed')
    parser.add_argument('-f', '--fpga-file', type=str, help='Specify the file, where the FPGA IPs are listed')
    parser.add_argument('-a','--master-address', type=str)
    parser.add_argument('-p','--master-port', type=str)
    
    args = parser.parse_args()


    host_file = args.host_file
    fpga_file = args.fpga_file
    comms = args.comms
    start_port = 5005
    
    global rank, size
    if args.master_address==None:
        args.master_address = "localhost"
    if args.master_port==None:
        args.master_port = "30505"
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port

    rank = mpi.Get_rank()
    size = mpi.Get_size()
    
    if size > 1:
        if (args.backend == 'mpi'):
            logger.debug("Starting mpi distributed")
            dist.init_process_group("mpi", rank=rank, world_size=size)
        else:
            logger.debug("Starting ACCL distributed")
            
            create_accl_process_group(args.simulator, args.comms, host_file, fpga_file)
            logger.debug("Start initialising PG")
            dist.init_process_group("ACCL", rank=rank, world_size=size)
            logger.debug("Finished initialising PG")
    else:
        logger.debug("starting local")
    device = 'cpu'

    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )

    if size > 1 : sampler = DistributedSampler
    else : sampler = lambda x : None
    #sampler=sampler(train_data))
    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                                              batch_size=128, 
                                              shuffle=False,
                                            ),
        'test'  : torch.utils.data.DataLoader(test_data, 
                                              batch_size=128, 
                                              shuffle=False),
    }
    #bucket_cap_mb=1
    cnn = CNN() 
    if size > 1 : cnn = DDP(cnn, bucket_cap_mb=1)
    #cnn.register_comm_hook(state=None, hook=sync_hook)
    loss_func = nn.CrossEntropyLoss()   
    num_epochs = 100
    profile = False
    
    schedule = torch.profiler.schedule(
        wait=0,
        warmup=0,
        active=200,
        repeat=1
    )

    if profile:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./accl_log/profiler_log'),
                record_shapes=True,
                with_stack=True
        ) as p:

            train(num_epochs, cnn, loaders, p)
            test()
    else:
        logger.debug("Start training without profiler")
        train(num_epochs, cnn, loaders)
        test()
        #shape = (28938,)
        #rand_torch = torch.rand(shape, dtype=torch.float32)
        #for i in range(2350):
            #if rank == 0:
                #time.sleep(0.1)
            #rand_torch = torch.rand(shape, dtype=torch.float32)
            #x = rand_torch.clone()
            #dist.all_reduce(x, dist.ReduceOp.SUM)

    if size > 1: 
        print("Dist.destroy_process_group")
        #Not sure which one needed but with no synchronisation destroy_process_group which should call ACCL PG destroy() could fail
        dist.barrier()
        mpi.Barrier()
        dist.destroy_process_group()
        
	   
	   
