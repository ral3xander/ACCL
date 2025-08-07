
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity

from datasets import load_dataset
from PIL import Image
import torch.cuda.nvtx as nvtx

import scipy

import numpy as np

from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import powerSGD_hook

times = []


    
def sync_hook(state, bucket):
    # Synchronize all ranks before all_reduce
    dist.barrier()
    # Proceed with default all_reduce
    return allreduce_hook(state, bucket)

def signSGD_hook(state, bucket):
   #We study signSGD with majority vote, where 1 bit is sent for each float (32 bit) leading to
    #32× compression. Majority vote operation is not associative thus requiring use of all-gather. Figure 7, shows
    #that despite signSGD being extremely quick to encode and decode, due to lack of comaptibility with all 
    #reduce communication time scales linearly. Further, due to overheads in creating buffers for the all-gather
    #primitive we can not scale signSGD on BERTBASE beyond 32 GPUs.

    #https://arxiv.org/pdf/1802.04434
    #https://arxiv.org/pdf/1810.05291

    
    return
def MSTopK_hook(sate, bucket):
    #Not compatible with allreduce
    #Determine most signifincat gradinets on every rank and gather them reconstrucing a dense gradinet matrix?
    #https://proceedings.mlsys.org/paper_files/paper/2021/file/25a3192c804d6b1c7d309c0155d3aa1a-Paper.pdf
    return

def evaluate_on_test_data(model, device, test_loader):
    model.eval()
    correct_num = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += len(labels)
            correct_num += (predicted == labels).sum().item()

    return correct_num / total


def train_epoch(model, dataloader, criterion, optimizer, device, bs, profiler=None):  # bs: bach size
    model.train()
    total_loss = 0
    total_correct = 0

    for batch in dataloader:
        if profiler: 
                profiler.step()
        images, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()  # reset gradients
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)

        correct_num = (predictions == labels).sum().item()
        total_correct += correct_num

    return total_loss / len(dataloader), total_correct / (len(dataloader) * bs)


def main():
    print("Lets get Started???")

    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    master = (os.environ["MASTER_ADDR"])
    port = int(os.environ["MASTER_PORT"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    

    print(f"global rank:", {global_rank}, " local_rank:", {local_rank}, "World size", {world_size},"local world size:", {local_world_size},"MASTER", {master}, ":",{port})
    print(f"Hello from local_rank {local_rank}, global_rank {global_rank}")
    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num_epochs", type=int, help="Number of training epochs.", default=1000
    )
    parser.add_argument(
        "--batch_size", type=int, help="Training batch size.", default=128
    )
    parser.add_argument(
        "--batch_size_scaled",
        action="store_true",
        help="Training batch size for one process.",
    )
    argv = parser.parse_args()

    # we need GPUs
    assert torch.cuda.is_available(), "DDP requires at least one GPU."
    # Initializes the default distributed process group, and this will also
    # initialize the distributed package.
    print("start init process group")
    dist.init_process_group(backend="nccl")
    print("end init process group")
    # training configuration
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    if argv.batch_size_scaled:
        print("argv.batch_size_scaled", argv.batch_size_scaled, num_epochs, batch_size)
        batch_size //= dist.get_world_size()
    else:
        print("argv.batch_size_scaled", argv.batch_size_scaled, num_epochs, batch_size)
    print(batch_size)
    learning_rate = 0.002
    model_dir = "saved_ddp_models"
    model_filename = "resnet_ddp.pth"
    log_every = 10

    model_filepath = os.path.join(model_dir, model_filename)

    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    master = (os.environ["MASTER_ADDR"])
    port = int(os.environ["MASTER_PORT"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    

    print(f"global rank:", {global_rank}, " local_rank:", {local_rank}, "World size", {world_size},"local world size:", {local_world_size},"MASTER", {master}, ":",{port})
    print(f"Hello from local_rank {local_rank}, global_rank {global_rank}")
    torch.cuda.set_device(local_rank)

    model_name = "resnet18"       # "resnet18", "resnet34", "resnet50"
    dataset_name = "cifar10"      # "cifar10", "cifar100", "imagenet"

    # Set dataset-specific parameters
    if dataset_name == "cifar10":
        num_classes = 10
        input_size = 32
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "cifar100":
        num_classes = 100
        input_size = 32
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "imagenet":
        num_classes = 1000
        input_size = 224
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.ImageNet(root='./data', split='train', transform=transform)
        test_dataset = torchvision.datasets.ImageNet(root='./data', split='val', transform=transform)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Select and modify model
    if model_name == "resnet18":
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif model_name == "resnet34":
        model = torchvision.models.resnet34(num_classes=num_classes)
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Adjust model for small inputs
    if input_size == 32:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    

    model = model.to(local_rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )
    
    #state = PowerSGDState(process_group, matrix_approximation_rank=1, start_powerSGD_iter=1000, min_compression_rate=2, use_error_feedback=True,
    #               warm_start=True, orthogonalization_epsilon=0, random_seed=0, 
    #               compression_stats_logging_frequency=10000, batch_tensors_with_same_shape=False)
    state = PowerSGDState(process_group=torch.distributed.group.WORLD, matrix_approximation_rank=4,start_powerSGD_iter=10, min_compression_rate=0.5)
    
    #ddp_model.register_comm_hook(state, powerSGD_hook)
    ddp_model.register_comm_hook(torch.distributed.group.WORLD, fp16_compress_hook)
    #ddp_model.register_comm_hook(state=None, hook=sync_hook)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
   



    # Sampler that restricts data loading to a subset of the dataset exclusive
    # to the current process
    #sampler=DistributedSampler(dataset=train_dataset),
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
    )

    # we will only test on rank 0
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=16
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9)

    log_epoch = 0
    # Training Loop
    training_start = time.perf_counter()
    for epoch in range(num_epochs):

        log_epoch += 1
        epoch_start_time = time.perf_counter()

        ddp_model.train()
        #train_loader.sampler.set_epoch(epoch)

        if epoch == 5 and local_rank == 0 and True:
            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=10, active=20, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./GPU_log/traces'),
                record_shapes=True,
                with_stack=True
            ) as prof:
                
                train_loss, train_acc = train_epoch(
                    ddp_model, train_loader, criterion, optimizer, local_rank, batch_size, prof
                )

        else:
            train_loss, train_acc = train_epoch(
                ddp_model, train_loader, criterion, optimizer, local_rank, batch_size
            )

        torch.cuda.synchronize()
        epoch_end_time = time.perf_counter()
        if epoch > 1 and global_rank == 0:
            times.append(epoch_end_time - epoch_start_time)
        if global_rank == 0:
            print(
                f"Epoch - {log_epoch}/{num_epochs}: time - {(epoch_end_time - epoch_start_time):.4f}s || loss_train - {train_loss:.4f} || accuracy_train - {train_acc:.4f}"
            )
            if log_epoch % log_every == 0:
                test_accuracy = evaluate_on_test_data(
                    model=ddp_model, device=local_rank, test_loader=test_loader
                )
                print(f"Accuracy on test dataset - {test_accuracy:.4f}")

    torch.cuda.synchronize()
    training_end = time.perf_counter()
    if global_rank == 0:
        print(f"Training took {(training_end - training_start):.4f} s")
        #torch.save(ddp_model.state_dict(), model_filepath)

    if global_rank == 0:
        average_time = np.mean(times)
        std_time = np.std(times, ddof=1)  # ddof=1 for sample standard deviation
    
        print(f"Average time last 7: {average_time:.4f} seconds")
        print(f"Standard deviation: {std_time:.4f} seconds")

    # Destroy the process group, and deinitialize the distributed package
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()



