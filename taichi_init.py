import taichi as ti
import pynvml
import os
from mpi4py import MPI


def device_init(hostnames):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # 在根节点上的
    if rank == 0:
        ti.init(arch=ti.cpu)
        return
    hostname = MPI.Get_processor_name()
    for i in range(len(hostnames)):
        if hostnames[i] == hostname:
            color = i
    # 获取当前计算卡在主机内的rank
    inner_comm = comm.Split(color)
    gpu_count = pynvml.nvmlDeviceGetCount()
    # 需要保证每个线程都有一块可以独占的gpu
    assert(inner_comm.rank < gpu_count)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(inner_comm.rank)
    os.environ['TI_VISIBLE_DEVICE'] = str(inner_comm.rank)
    
    ti.init(arch=ti.cuda)
    