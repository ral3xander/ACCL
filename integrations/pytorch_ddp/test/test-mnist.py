import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from mpi4py.MPI import COMM_WORLD as mpi

import argparse
import os
import sys
import logging
import time

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logger = logging.getLogger(__name__)

if "ACCL_DEBUG" in os.environ and os.environ["ACCL_DEBUG"]=="1":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)

logger.debug(f"Python executable: {sys.executable}")

def create_accl_process_group(simulator, comms, host_file, fpga_file):
    logger.debug("creating accl process group")
    #rxbufsize = 4096 * 1024
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
        rxbufsize = 4096 * 40
        ranks = [accl.Rank("127.0.0.1", 5500 + i, i, rxbufsize) for i in range(size)]

    if comms == 'udp':
        design = accl.ACCLDesign.udp
    elif comms == 'tcp':
        design = accl.ACCLDesign.tcp
    elif comms == 'cyt_rdma': # and not simulator:
        design = accl.ACCLDesign.cyt_rdma

    accl.create_process_group(ranks, design, bufsize=rxbufsize, initialize=True, simulation=args.simulator)


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

def train(num_epochs, cnn, loaders):

    start_time_train = time.perf_counter()
    
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])

    optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            p.step()
            if (i-1) % 100 == 0 or i == 0: start_time = time.perf_counter()
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
            
            if (i+1) % 100 == 0:
            
                end_time = time.perf_counter()
                measured_time = (end_time - start_time)
                logger.debug ('rank: {} Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time(s): {}' 
                       .format(mpi.Get_rank(), epoch + 1, num_epochs, i + 1, total_step, loss.item(), measured_time))

    end_time_train = time.perf_counter()
    measured_time_train = (end_time_train - start_time_train)

    print('Total train time: ' + str(measured_time_train))
        
def test():
    # Test the model
    start_time_test = time.perf_counter()
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            p.step()
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct_current = (pred_y == labels).sum().item()
            total += labels.size(0)
            correct += correct_current
            
            print(f'Test Batch accuracy: {correct_current}/{labels.size(0)} {correct_current/float(labels.size(0))}')


    end_time_test = time.perf_counter()
    measured_time_test = (end_time_test - start_time_test)

    print('Total test time: ' + str(measured_time_test))            
    print(f'Total accuracy: {correct}/{total} {correct/float(total)}')
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("-d", type=bool, default=None)

    parser.add_argument('-b', '--backend', choices=['accl', 'mpi'], default='mpi', help='Chose backend for collectives, either accl with fpga hardware support or software mpi')
    parser.add_argument('-s', '--simulator', action='store_true', default=False, help='Use simulation instead of hardware')
    parser.add_argument('-c', '--comms', choices=['udp', 'tcp', 'cyt_rdma', 'mpi'], default='cyt_rdma', help='Run tests over specified communication backend')
    
    parser.add_argument('-i', '--host-file', type=str, help='Specify the file, where the host IPs are listed')
    parser.add_argument('-f', '--fpga-file', type=str, help='Specify the file, where the FPGA IPs are listed')
    parser.add_argument('-a','--master-address', type=str)
    parser.add_argument('-p','--master-port', type=str)
    
    args = parser.parse_args()


    if args.n == 1 and args.d == None :
        print("only one machine specified. Assuming Non distributed setup")
        args.d = False
    elif args.n > 1 and args.d == None:
        print("Assung DDP setup")
        args.d = True


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

    if args.d:
        if (args.backend == 'mpi'):
            logger.debug("Starting mpi distributed")
            dist.init_process_group("mpi", rank=rank, world_size=size)
        else:
            logger.debug("Starting ACCL distributed")
            import accl_process_group as accl
            create_accl_process_group(args.simulator, args.comms, host_file, fpga_file)
            dist.init_process_group("ACCL", rank=rank, world_size=size)
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

    if args.d : sampler = DistributedSampler
    else : sampler = lambda x : None
    
    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                                              batch_size=100, 
                                              shuffle=False,
                                              sampler=sampler(train_data)),
        'test'  : torch.utils.data.DataLoader(test_data, 
                                              batch_size=100, 
                                              shuffle=False,
                                              sampler=sampler(test_data)),
    }

    cnn = CNN()
    if args.d : cnn = DDP(cnn, bucket_cap_mb=1)

    loss_func = nn.CrossEntropyLoss()   

    num_epochs = 3

    print("starting training")
    logger.debug("starting training")
    logger.debug(rank)
    logger.debug(size)
    print(rank)
    print(size)
    
    schedule = torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=10,
        repeat=3
    )
    if True:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./accl_log/profiler_log'),
                record_shapes=True,
                with_stack=True
        ) as p:

        
            train(num_epochs, cnn, loaders)
            test()


    if args.d : dist.destroy_process_group()
