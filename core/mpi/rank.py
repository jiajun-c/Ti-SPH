from mpi4py import MPI

class MPIrank:
    def __init__(self, hosts) -> None:
        host = MPI.Get_processor_name()
        self.comm = MPI.Comm
        self.GlobalRank = self.comm.Get_rank()
        self.size = self.comm.size
        for i in range(len(hosts)):
            if hosts[i] == host:
                self.LocalRank = i
         
        