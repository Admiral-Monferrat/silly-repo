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
        # Assuming that we want to return a new Communicator instance wrapping the new communicator.
        new_comm = self.comm.Split(key=key, color=color)
        new_instance = Communicator()
        new_instance.init(new_comm)
        return new_instance

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
        rank = self.Get_rank()
        size = self.Get_size()
        # Number of bytes in the entire array.
        bytes_count = src_array.itemsize * src_array.size

        if rank == 0:
            # Start with a copy of the root's own data.
            result = np.copy(src_array)
            # Receive data from all other processes and reduce.
            for i in range(1, size):
                temp = np.empty_like(src_array)
                self.comm.Recv(temp, source=i, tag=0)
                if op == MPI.SUM:
                    result += temp
                elif op == MPI.PROD:
                    result *= temp
                elif op == MPI.MAX:
                    result = np.maximum(result, temp)
                elif op == MPI.MIN:
                    result = np.minimum(result, temp)
                else:
                    raise NotImplementedError("Reduction operation not implemented")
            # Copy the final result to the destination array.
            dest_array[:] = result
            # Broadcast the result to all other processes.
            for i in range(1, size):
                self.comm.Send(result, dest=i, tag=0)
            # Update transfer cost for root.
            self.total_bytes_transferred += 2 * (size - 1) * bytes_count
        else:
            # Non-root processes send their data to root...
            self.comm.Send(src_array, dest=0, tag=0)
            # ... and then receive the reduced result.
            self.comm.Recv(dest_array, source=0, tag=0)
            # Update transfer cost for non-root processes.
            self.total_bytes_transferred += 2 * bytes_count

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
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()
        # Ensure arrays are divisible evenly.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )
        seg_size = src_array.size // nprocs

        for j in range(nprocs):
            if j == rank:
                # For the local segment, perform a direct copy.
                dest_array[j*seg_size:(j+1)*seg_size] = src_array[j*seg_size:(j+1)*seg_size]
            else:
                # Calculate bytes for this segment.
                send_seg_bytes = src_array[j*seg_size:(j+1)*seg_size].nbytes
                recv_seg_bytes = dest_array[j*seg_size:(j+1)*seg_size].nbytes
                self.total_bytes_transferred += send_seg_bytes + recv_seg_bytes
                # Exchange segments with process j.
                self.comm.Sendrecv(
                    sendbuf=src_array[j*seg_size:(j+1)*seg_size],
                    dest=j,
                    recvbuf=dest_array[j*seg_size:(j+1)*seg_size],
                    source=j,
                    tag=0
                )
