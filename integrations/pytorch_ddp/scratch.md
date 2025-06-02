Doucment for my personal notes.

## Collectives in ProcessGroupACCL Emulation:

### Supported
| Collective       | MPI Sidestep | Notes |
|------------------|:------------:|-------| 
| `broadcast`      | ✅ | Works
| `allreduce`      | ✅ |
| `reduce`         | ❌ |
| `allgather`      | ✅ |
| `gather`         | ✅ |
| `scatter`        | ✅ |
| `alltoall_base`  | ❌ |
| `send`           | ❌ |
| `receive`        | ❌ |

### Not supported
| Collective           | Notes |
|----------------------|-------|
| `allreduce_coalesced` |  |
| `allgather_coalesced` |  |
| `reduce_scatter`      |  |
| `alltoall`            |  |
| `recvAnysource`       |  |
| `allgather_base`      |  |



### Broadcast

| Ranks | NBuf | BufSize | Message | Works? | Notes |
|-------|------|---------|-------- |--------|-------|
| 2 | 1 | 1023 | 2048 * 4 | ❌ |  mismatch |
| 2 | 1 | 1024 | 2048 * 4 | ✅ |  mismatch |


changed segmentation from MAX_SEGMENT_SIZE to bufsize. fixed issues except: see above

### Allreduce

| Ranks | NBuf | BufSize | Message | Works? | Notes |
|-------|------|---------|-------- |--------|-------|
| 2 | 1 | 1023 | 2048 * 4 | ❌ |  stuck |
| 2 | 1 | 1024 | 2048 * 4 | ❌ |  stuck |



### Reduce

| 2 | 1 | 1023 | 2048 * 4 | ✅ |   |
| 2 | 1 | 1024 | 2048 * 4 | ❌✅ |  completed sucessfully but: YOUR APPLICATION TERMINATED WITH THE EXIT STRING: Aborted (signal 6) |




### Gather/AllGather

Currently sidestepped!

### Scatter

Works so far

### AllToAll

Works so far (Unless input tensor cannot be divided evenly accross the ranks which is ok I think, not sure though whether custom backend alltoall is acutally called or just mpi sidestep. Gotta have a look.)

### Send/Receive

Works so far



## Collectives in ProcessGroupACCL hardware:




## Stuff

python setup.py build_ext --inplace
cp /mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/accl_process_group/_c/ProcessGroupACCL.cpython-310-x86_64-linux-gnu.so /mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/envACCL/lib/python3.10/site-packages/accl_process_group/_c/

source /mnt/scratch/ralexander/ACCL/test/host/Coyote/run_scripts/program_hacc_localv2.sh && sudo /opt/hdev/cli/program/fpga_chmod 0

bash /mnt/scratch/ralexander/ACCL/test/host/Coyote/run_scripts/run.sh




mpirun -n 2 -f ./accl_log/host bash -c "source /mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/../venv/bin/activate && source /mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/../setup.sh && python /mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/test-generic.py -c cyt_rdma -i ./accl_log/host -f ./accl_log/fpga -a 10.253.74.90 -p 30505"

mpirun -n 2 -f ./accl_log/host -outfile-pattern "/mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/accl_log/rank_%r_stdout" -errfile-pattern "/mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/accl_log/rank_%r_stderr" bash -c "source /mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/../venv/bin/activate && source /mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/../setup.sh && python /mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/test-generic.py -c cyt_rdma -i ./accl_log/host -f ./accl_log/fpga -a 10.253.74.90 -p 30505" &


export MPIEXEC_PREFIX_DEFAULT=1

enp65s0f0np0

mpirun -n 1 -f ./accl_log/host --iface enp65s0f0np0 bash -c "source /mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/../envACCL/bin/activate && source /mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/../setup.sh && python /mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/test-generic.py -c cyt_rdma -i ./accl_log/host -f ./accl_log/fpga -a 10.253.74.74 -p 30505"




export MPICC=/mnt/scratch/zhe/mpich/install/lib/mpicc
export MPICXX=/mnt/scratch/zhe/mpich/install/lib/mpicxx
pip install --no-binary=mpi4py mpi4py



DEBUGGING ACCL allreduce:


Single rank, cyt_rdma, to/from_fpga false, async false

tensor (first 10): [0.4775, 0.1798, 0.2428, 0.2166, 0.9245, 0.6069, 0.5380, 0.7022, 0.7608, 0.6129]
send (first 10):   [0.477509, 0.179843, 0.242757, 0.216641, 0.924459, 0.606868, 0.537952, 0.702206, 0.760769, 0.612871]
receive buffer (first 10): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

accl->allreduce...
copy receive_buffer to in_tensor...

in_tensor (first 10): [0.1987 0.1433 0.3335 0.8004 0.9047 0.8434 0.6618 0.8133 0.8181 0.4208]
send buffer (first 10): [0.477509, 0.179843, 0.242757, 0.216641, 0.924459, 0.606868, 0.537952, 0.702206, 0.760769, 0.612871]
receive buffer (first 10): [0.198747, 0.143286, 0.333521, 0.800422, 0.90471, 0.843372, 0.661814, 0.813258, 0.818123, 0.420764]


