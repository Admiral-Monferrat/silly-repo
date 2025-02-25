from mpi4py import MPI
import numpy as np

class Communicator(object):
    def init(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return Communicator(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()
        assert src_array.size % nprocs == 0, "src_array size must be divisible by the number of processes"
        assert dest_array.size % nprocs == 0, "dest_array size must be divisible by the number of processes"
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)
        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        root = 0
        nprocs = self.Get_size()
        rank = self.Get_rank()
        src_byte = src_array.itemsize * src_array.size
        
        if rank == root:
            self.comm.Reduce(src_array, dest_array, op=op, root=root)
        else:
            self.comm.Reduce(src_array, None, op=op, root=root)
        
        self.comm.Bcast(dest_array, root=root)
        
        if rank == root:
            self.total_bytes_transferred += 2 * (nprocs - 1) * src_byte
        else:
            self.total_bytes_transferred += 2 * src_byte

    def myAlltoall(self, src_array, dest_array):
        nprocs = self.Get_size()
        rank = self.Get_rank()
        
        assert src_array.size % nprocs == 0, "src_array size must be divisible by the number of processes"
        assert dest_array.size % nprocs == 0, "dest_array size must be divisible by the number of processes"
        
        seg_size_src = src_array.size // nprocs
        send_seg_bytes = src_array.itemsize * seg_size_src
        seg_size_dest = dest_array.size // nprocs
        recv_seg_bytes = dest_array.itemsize * seg_size_dest
        
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)
        
        for k in range(nprocs):
            src_start = k * seg_size_src
            src_end = src_start + seg_size_src
            send_segment = src_array[src_start:src_end]
            
            dest_start = k * seg_size_dest
            dest_end = dest_start + seg_size_dest
            recv_segment = dest_array[dest_start:dest_end]
            
            if k == rank:
                np.copyto(recv_segment, send_segment)
            else:
                self.comm.Sendrecv(send_segment, dest=k, recvbuf=recv_segment, source=k)