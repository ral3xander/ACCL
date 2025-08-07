import numpy as np
import matplotlib.pyplot as plt



import json
import pandas as pd

# Load the JSON file
f = "../GPU_log/traces/resnet18_Cifar10_Rank2_sync.json"
#f = "../GPU_log/traces/resnet18_Cifar10_Rank2.json"
#f = "../GPU_log/traces/resnet18_Cifar10_Rank4_sync.json"
#f = "../GPU_log/traces/resnet18_Cifar10_Rank4.json"
dataset = "Cifar10"
model = "ResNet18"
ranks = "4"

with open(f, 'r') as f:

    trace_data = json.load(f)

if isinstance(trace_data, dict) and "traceEvents" in trace_data:
    trace_data = trace_data["traceEvents"]


# Convert to DataFrame
df = pd.DataFrame(trace_data)

# Keep relevant fields
df = df[['name', 'ts', 'dur', 'args']]



def plot_hist(name, data, bins, path):
    
    times = np.array( data)
    plt.figure(figsize=(10, 6))
    plt.hist(times, bins=bins, edgecolor='black')
    plt.title(name)
    plt.xlabel('Execution Time (us)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_scatter(name, data, path):
    x = range(len(data))
    plt.figure(figsize=(8, 4))
    plt.scatter(x, data)
    plt.title(name)
    plt.xlabel("Index")
    plt.ylabel("Time (us)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def print_stats(name, data):
    data_np = np.array(data)
    mean = np.mean(data_np)
    median = np.median(data_np)
    stdev = np.std(data_np)
    print(f"\nGroup {name}:")
    print(f"  Count       : {len(data)}")
    print(f"  Mean ± Std  : {mean:.2f} ± {stdev:.2f} us")
    print(f"  Median      : {median:.2f} us")






def parse_sync(f):
    with open(f, 'r') as f:
        trace_data = json.load(f)
    if isinstance(trace_data, dict) and "traceEvents" in trace_data:
        trace_data = trace_data["traceEvents"]


    df = pd.DataFrame(trace_data)
    df = df[['name', 'ts', 'dur', 'args']]
    operation = "ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
    matched = df[df["name"] == operation]
    matched = matched.iloc[1::2].reset_index(drop=True)
    groups = {1: [], 2: [], 3: []}
    for i, row in matched.iterrows():
        group_id = (i % 3) + 1
        groups[group_id].append(row["dur"])

    return groups

def parse_normal(f):
    with open(f, 'r') as f:
        trace_data = json.load(f)
    if isinstance(trace_data, dict) and "traceEvents" in trace_data:
        trace_data = trace_data["traceEvents"]
    df = pd.DataFrame(trace_data)
    df = df[['name', 'ts', 'dur', 'args']]
    operation = "ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
    matched = df[df["name"] == operation]
    matched = matched.reset_index()
    groups = {1: [], 2: [], 3: []}
    for i, row in matched.iterrows():
        group_id = (i % 3) + 1
        groups[group_id].append(row["dur"])

    return groups



def plot_ResNet18_cifar(colors=None, title="ResNet18 on CIFAR-10 Bucket 3"):
    # File paths
    f1 = "../GPU_log/traces/resnet18_Cifar10_Rank2_sync.json"
    f2 = "../GPU_log/traces/resnet18_Cifar10_Rank2.json"
    f3 = "../GPU_log/traces/resnet18_Cifar10_Rank4_sync.json"
    f4 = "../GPU_log/traces/resnet18_Cifar10_Rank4.json"


    # Dataset dictionary
    datasets = {
        "rank2_sync": parse_sync(f1)[3],
        "rank2": parse_normal(f2)[3],
        "rank4_sync": parse_sync(f3)[3],
        "rank4": parse_normal(f4)[3]
    }
    print(datasets)
    num_datasets = len(datasets)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if colors is None:
        colors = default_colors[:num_datasets]

    plt.figure(figsize=(10, 6))

    # Scatter plot
    for i, (label, data) in enumerate(datasets.items()):
        print(len(data))
        x = np.arange(len(data))
        plt.scatter(x, data, color=colors[i % len(colors)], alpha=0.7)

    
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Time (us)")
    plt.subplots_adjust(bottom=0.3)

    # Summary stats at the bottom
    for i, (label, data) in enumerate(datasets.items()):
        mean = np.mean(data)
        std = np.std(data)
        median = np.median(data)

        x_pos = 0.2 + i * 0.2
        y_pos = 0.1

        plt.figtext(
            x_pos, y_pos,
            f"{label}\nμ±σ = {mean:.2f}±{std:.2f}\nmedian = {median:.2f}",
            ha="center",
            fontsize=9,
            bbox=dict(facecolor=colors[i % len(colors)], alpha=0.3, boxstyle="round,pad=0.3")
        )
    
    plt.savefig("./plotsCuda//Resnet18_Cifar10_bucket3")


plot_ResNet18_cifar()