import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import statistics
import os
import json
import numpy as np
import math

def parse_Mnist(folder, backend, test, ranks, log, epochs, numBatches):
    comm_pattern = re.compile(r"(.*)_(.*)_(.*) durationUs: (.*)")
    training_pattern = re.compile(r"Epoch \[(\d+)/(\d+)\], Loss: ([\d\.]+), Time\(s\): ([\d\.]+)")
    train_time_pattern = re.compile(r"Total train time: ([\d\.]+)")
    accuracy_pattern = re.compile(r"Total accuracy:\s*(\d+)/(\d+)\s+([\d\.]+)")
    segmentation_pattern = re.compile(r"\[(\w+)\] Segmenting tensor of size (\d+) into (\d+)-sized elements")
    Mnist = {
            "total_time":-1,
            "accuracy":-1,
            "data": {
                rank: {
                    epoch: {
                        "time": 0,
                        "batches": {
                            batch: []
                            for batch in range(numBatches)
                        }

                    }
                    for epoch in range(epochs)
                }
                for rank in range(ranks)
            }
    }   
    for rank in range(ranks):
        filepath = os.path.join(folder, f"rank_{rank}_stderr")
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            continue

        
    
        epoch = 0
        batch = 0
       
        call = 0
        seg_entry = {}
        call_entry = {}
        with open(filepath, 'r') as f:
            for line in f:
                
                match = training_pattern.search(line)
                if match:
                    epoch_current, epoch_total, loss, time = match.groups()
                    epoch_current = int(epoch_current)
                    epoch_total = int(epoch_total)
                    loss = float(loss)
                    time = float(time)
                    Mnist["data"][rank][epoch]["time"] = time
                    epoch += 1
                    batch = 0
                   

                match = comm_pattern.search(line)
                if match:
                    part, op, cnt_str, duration_str = [m.strip() for m in match.groups()]
    
                    
                    if part == "lock":
                        seg_entry = {
                            "op": op, 
                            "size": int(cnt_str),
                            "lock": float(duration_str),
                            "lib": -1,
                            "total": -1
                        }
                        call_entry = {
                            "op": op,
                            "size": int(cnt_str),
                            "segments": [],
                            "total": -1
                        }
                        if backend == 'accl':
                            seg_entry = {
                                "op": op, 
                                    "size": int(cnt_str),
                                    "lock": float(duration_str),
                                    "init": -1,
                                    "device": -1,
                                    "lib": -1,
                                    "copy": -1,
                            }
                            call_entry = {
                                "op": op,
                                "size": int(cnt_str),
                                "segments": [],
                                "total": -1
                            }


                    elif part == "total":
                        seg_entry["total"] = float(duration_str)
                        call_entry["segments"].append(seg_entry)
                        call_entry["total"] = float(duration_str)
                        Mnist["data"][rank][epoch]["batches"][batch].append(call_entry)
                        batch += 1

                    else:
                        seg_entry[part] = float(duration_str)
                
                match = accuracy_pattern.search(line)
                if match:
                    left, right, percent = [m.strip() for m in match.groups()]
                    left = int(left)
                    right = int(right)
                    percent = float(percent)
                    Mnist["accuracy"] = percent
                
                match = train_time_pattern.search(line)
                if match:
                    [total_train_time] = [m.strip() for m in match.groups()]
                    Mnist["total_time"] = total_train_time
    return Mnist
    


def parse_resNet18(folder, backend, ranks):
    comm_pattern = re.compile(r"(.*)_(.*)_(.*) durationUs: (.*)")
    training_pattern = re.compile(r"Epoch \[(\d+)/(\d+)\], Loss: ([\d\.]+), Time\(s\): ([\d\.]+)")
    train_time_pattern = re.compile(r"Total train time: ([\d\.]+)")
    accuracy_pattern = re.compile(r"Total accuracy:\s*(\d+)/(\d+)\s+([\d\.]+)")
    segmentation_pattern = re.compile(r"\[(\w+)\] Segmenting tensor of size (\d+) into (\d+)-sized elements")
    call_pattern = re.compile(r"\bcall run_\w+\b")

    epoch = 0
    batch = 0
    nsegs = 0
    seg_entry = {}
    call_entry = {}
    resNet18 = {
            "total_time":-1,
            "accuracy":-1,
            "data": {
                rank: {
                    epoch: {
                        "time": 0,
                        "batches": {0: []
                        }

                    }
                }
                for rank in range(ranks)
            }
    }

    for rank in range(ranks):
        filepath = os.path.join(folder, f"rank_{rank}_stderr")
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            continue
        epoch = 0
        batch = 0
        seg_entry = {}
        call_entry = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                #print(line)
                match = segmentation_pattern.search(line)
                if match:
                    op, cnt_str, segment_size = match.groups()
                    
                    
                    
                    if int(cnt_str) == 9461800 and backend == 'accl':
                        batch += 1
                        resNet18["data"][rank][epoch]["batches"][batch] = []
                    if int(cnt_str) == 4292648 and backend == 'accl':
                        batch += 1
                        resNet18["data"][rank][epoch]["batches"][batch] = []
                    
                    call_entry = {
                            "op": op,
                            "size": int(cnt_str),
                            "segments": [],
                            "total": -1
                    }
                    
                  

                match = training_pattern.search(line)
                if match:
                    epoch_current, epoch_total, loss, time = match.groups()
                    epoch_current = int(epoch_current)
                    epoch_total = int(epoch_total)
                    loss = float(loss)
                    time = float(time)
                    resNet18["data"][rank][epoch]["time"] = time
                    epoch += 1
                    batch = 0
                    resNet18["data"][rank][epoch] = {"time":-1, "batches":{0:[]}}

                match = comm_pattern.search(line)
                if match:
                    part, op, cnt_str, duration_str = [m.strip() for m in match.groups()]
                   
                    if part == "lock":
                        seg_entry = {
                            "op": op, 
                            "size": int(cnt_str),
                            "lock": float(duration_str),
                            "lib": -1,
                        }

                        if backend == 'accl':
                            seg_entry = {
                                "op": op, 
                                "size": int(cnt_str),
                                "lock": float(duration_str),
                                "init": -1,
                                "device": -1,
                                "lib": -1,
                                "copy": -1,
                            }
                        if call_entry == {}:
                             call_entry = {
                                "op": op,
                                "size": int(cnt_str),
                                "segments": [],
                                "total": -1
                            }

                        else:
                            nsegs -=1
                    
                    if part == "copy" and backend == 'accl':
                        seg_entry[part] = float(duration_str)
                        call_entry["segments"].append(seg_entry)
                    if part == "lib" and backend == 'mpi':
                        seg_entry[part] = float(duration_str)
                        call_entry["segments"].append(seg_entry)
                        if int(cnt_str) == 8739072:
                            batch += 1
                            resNet18["data"][rank][epoch]["batches"][batch] = []
                        if int(cnt_str) == 4292648:
                            batch += 1
                            resNet18["data"][rank][epoch]["batches"][batch] = []


                    elif part == "total":
                        call_entry["total"] = float(duration_str)
                        resNet18["data"][rank][epoch]["batches"][batch].append(call_entry)
                        call_entry = {}
                       

                    elif part in seg_entry :
                        seg_entry[part] = float(duration_str)
                
                match = accuracy_pattern.search(line)
                if match:
                    left, right, percent = [m.strip() for m in match.groups()]
                    left = int(left)
                    right = int(right)
                    percent = float(percent)
                    resNet18["accuracy"] = percent
                
                match = train_time_pattern.search(line)
                if match:
                    [total_train_time] = [m.strip() for m in match.groups()]
                    resNet18["total_time"] = total_train_time
    return resNet18
    




# PARAMETERS
#log_dir = "../accl_log/Accl_Rank2_generic_0107_1057/"

#log_dir = "../accl_log/Mpi_Rank2_Mnist_Batch128_Epoch10_0607_1600/"
#log_dir = "../accl_log/Mpi_Rank2_Mnist_Batch128_Epoch10_0607_1700/"
#log_dir = "/mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/accl_log/Mnist/Mpi_Rank2_Mnist_Batch128_Epoch1_Bucket1_0807_1200"

#log_dir = "../accl_log/Accl_Rank2_Mnist_Batch128_Epoch10_0107_0958/"
#log_dir = "../accl_log/Accl_Rank2_Mnist_Batch128_Epoch10_0607_1600/"
#log_dir = "../accl_log/Accl_Rank2_Mnist_Batch128_Epoch10_0607_1700/"


#log_dir ="/mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/accl_log/Mnist/Final/Mpi2"

#log_dir = "/mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/accl_log/resnet18/Accl_Rank2_resnet18Cifar_Batch128_Epoch10_almost"

log_dir = "/mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/accl_log/resnet18/Final/Inspection"

#log_dir= "/mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/accl_log/resnet18/Final/Mpi2"

if False:

    Mnist = parse_Mnist(log_dir, "mpi", "mnist", 2, True, 13, 469)

    with open("mnist_output.json", "w") as f:
        json.dump(Mnist, f, indent=2)


log_dir= "/mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/accl_log/resnet18/Final/Mpi2"
ResnetMpi = parse_resNet18(log_dir, "mpi", 2)

with open("resnetMpi_output.json", "w") as f:
    json.dump(ResnetMpi, f, indent=2)

log_dir = "/mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/accl_log/resnet18/Final/Inspection2"
ResnetAccl = parse_resNet18(log_dir, "accl", 2)

with open("resnetAccl_output.json", "w") as f:
    json.dump(ResnetAccl, f, indent=2)


log_dir = "/mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/accl_log/resnet50/Final/Inspection"
Resnet50Accl = parse_resNet18(log_dir, "accl", 2)

with open("resnet50Accl_output.json", "w") as f:
    json.dump(Resnet50Accl, f, indent=2)

log_dir = "/mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/accl_log/resnet50/Final/Mpi2"
Resnet50Mpi = parse_resNet18(log_dir, "mpi", 2)

with open("resnet50Mpi_output.json", "w") as f:
    json.dump(Resnet50Mpi, f, indent=2)



def extract_per_batch_times(cnn):
    times = []
    for b in cnn["data"][0][3]["batches"]:
        t = 0
        for c in cnn["data"][0][3]["batches"][b]:
            if c["op"] == "Allreduce":
                t += c["total"]
        times.append(t)
    return times




def bucket_times(cnn):
    times = {}
    for b in cnn["data"][0][3]["batches"]:
        for c in cnn["data"][0][3]["batches"][b]:
            if c["op"] == "Allreduce":
                if c["size"] in times:
                    times[c["size"]].append(c["total"])
                else:
                    times[c["size"]] = [c["total"]]
    return times
times = bucket_times(ResnetAccl)
times1 = extract_per_batch_times(ResnetAccl)
times2 = extract_per_batch_times(ResnetMpi)
times3 = extract_per_batch_times(Resnet50Accl)
times4 = extract_per_batch_times(Resnet50Mpi)
sizes = list(times.keys())
means = [np.mean(times) for times in times.values()]
stds = [np.std(times) for times in times.values()]
print(sizes)
print(means)
print(stds)

def plot_hist(name, data, bins, rank_id, filter=True):
    data = data [1000:1100]
    times = np.array(data[100:1100] if len(data) > 1100 else data)
    if filter:
        lower, upper = np.percentile(times, [0, 100])
        times = times[(times >= lower) & (times <= upper)]

    plt.figure(figsize=(10, 6))
    plt.hist(times, bins=bins, edgecolor='black')
    plt.title(name)
    plt.xlabel('Execution Time (us)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(name)
    plt.close()

def plot_scatter(name, data):
    data = data [1:]
    x = range(len(data))
    plt.figure(figsize=(8, 4))
    plt.scatter(x, data)
    plt.title(name)
    plt.xlabel("Index")
    plt.ylabel("Time (us)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(name + ".png")
    plt.close()
def plot_bar_chart(means, stds, labels=None, title="Bar", ylabel="Value"):

    if labels is None:
        labels = [str(i) for i in range(len(means))]
    
    x_pos = np.arange(len(means))
    
    plt.figure(figsize=(8, 6))
    plt.bar(x_pos, means, yerr=stds, capsize=5)
    plt.xticks(x_pos, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(x_pos, labels, rotation=90, ha='right') 
    plt.tight_layout()
    plt.savefig("bar.png")

plot_bar_chart(means, stds, sizes)

plot_scatter("18AcclScatter", times1)
#plot_scatter("18MpiScatter", times2)
#plot_scatter("50AcclScatter", times3)
#plot_scatter("50MpiScatter", times4)
                





