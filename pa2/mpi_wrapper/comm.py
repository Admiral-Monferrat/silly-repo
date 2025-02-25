from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
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
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
        - For non-root processes: one send and one receive.
        - For the root process: (n-1) receives and (n-1) sends.
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        root = 0
        
        # Calculate the number of bytes transferred
        array_bytes = src_array.itemsize * src_array.size
        
        # Copy source data to destination array for the root process
        if rank == root:
            np.copyto(dest_array, src_array)
            temp_array = np.empty_like(src_array)
            
            # Root receives and reduces data from all other processes
            for i in range(1, size):
                self.comm.Recv(temp_array, source=i)
                if op == MPI.SUM:
                    dest_array += temp_array
                elif op == MPI.PROD:
                    dest_array *= temp_array
                elif op == MPI.MAX:
                    np.maximum(dest_array, temp_array, dest_array)
                elif op == MPI.MIN:
                    np.minimum(dest_array, temp_array, dest_array)
                
                # Update bytes transferred (receive from each process)
                self.total_bytes_transferred += array_bytes
            
            # Root broadcasts the result to all other processes
            for i in range(1, size):
                self.comm.Send(dest_array, dest=i)
                
                # Update bytes transferred (send to each process)
                self.total_bytes_transferred += array_bytes
        else:
            # Non-root processes send their data to root
            self.comm.Send(src_array, dest=root)
            
            # Non-root processes receive the result from root
            self.comm.Recv(dest_array, source=root)
            
            # Update bytes transferred (one send and one receive per non-root process)
            self.total_bytes_transferred += 2 * array_bytes
        
    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
        - For the local segment (when destination == self), a direct copy is done.
        - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        
        # Ensure arrays can be evenly partitioned
        assert src_array.size % size == 0, "src_array size must be divisible by the number of processes"
        assert dest_array.size % size == 0, "dest_array size must be divisible by the number of processes"
        
        # Calculate segment size
        segment_size = src_array.size // size
        
        # Calculate bytes per segment
        segment_bytes = src_array.itemsize * segment_size
        
        # Create a temporary buffer for receiving data
        temp_buffer = np.empty(segment_size, dtype=src_array.dtype)
        
        for i in range(size):
            # Calculate the segment indices for source and destination arrays
            src_start = i * segment_size
            src_end = src_start + segment_size
            dest_start = rank * segment_size
            dest_end = dest_start + segment_size
            
            if i == rank:
                # For local segment, just copy the data directly
                dest_array[src_start:src_end] = src_array[src_start:src_end]
            else:
                # For remote segments, exchange data using Sendrecv
                self.comm.Sendrecv(
                    src_array[src_start:src_end], dest=i, 
                    recvbuf=temp_buffer, source=i
                )
                
                # Copy received data to the appropriate segment of dest_array
                dest_array[i*segment_size:(i+1)*segment_size] = temp_buffer
                
                # Update bytes transferred (each process sends one segment and receives one segment)
                self.total_bytes_transferred += 2 * segment_bytes
