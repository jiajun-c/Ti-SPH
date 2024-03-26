import taichi as ti
import trimesh as trim
import numpy as np
from mpi4py import MPI
from functools import reduce
from core.mpi.rank import MPIrank
from paritcle_system_base import ParticleSystemBase

def calc_ncols_from_rank(rank, size, NCOLS):
    ncols = int(NCOLS / size)       
    if ((NCOLS % size) != 0):
        if (rank == size - 1):
            ncols += NCOLS % size
    return ncols


@ti.data_oriented
class ParticleSystemMPI(ParticleSystemBase):
    def __init__(self, simulation_config) -> None:
        # mpi info
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.size
        self.global_rank = self.comm.rank
        self.gpu_node_rank = self.global_rank - 1
        self.gpu_node_size = self.size-1
        self.simulation_config = simulation_config
        self.configuration = self.simulation_config['configuration']
        self.rigidBodiesConfig = self.simulation_config['rigidBodies']  # list
        self.fluidBlocksConfig = self.simulation_config['fluidBlocks']  # list
        self.density0 = self.configuration['density0']
        
        self.dim = self.configuration['dim']
        
        self.domain_start = np.array(self.configuration['domainStart'])
        self.domain_end = np.array(self.configuration['domainEnd'])
        self.domain_size = self.domain_end - self.domain_start
        self.material_boundary = 0
        self.material_fluid = 1
        
        self.fluid = self.simulation_config['fluidBlocks']
        self.rigid = self.simulation_config['rigidBodies']
        gapx = calc_ncols_from_rank(self.gpu_node_rank, self.gpu_node_size, self.grid_num[0])
        if self.gpu_node_rank > 0:
            gapx += 1
        if self.gpu_node_rank < self.gpu_node_size - 1:
            gapx += 1
        self.local_grid_num = np.array([gapx, self.grid_num[1], self.grid_num[2]])
        self.local_grid_count = int(reduce(lambda x, y : x*y, self.local_grid_num))
        self.local_grid_particle_num = ti.field(ti.i32, shape=int(reduce(lambda x, y : x*y, self.local_grid_num)))
        self.local_grid_particle_num_temp = ti.field(ti.i32, shape=int(reduce(lambda x, y : x*y, self.local_grid_num)))

        self.particle_radius = self.configuration['particleRadius']
        self.particle_diameter = 2 * self.particle_radius
        self.support_length = 4.0 * self.particle_radius
        self.padding = self.support_length  # padding is used for boundary condition when particle collide with wall
        # self.particle_num = ti.field(int, shape=())
        # self.particle_max_num = 0
        # 获取起始的index和结束的index
        self.get_range_grid_index(self)
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

    # 0 1 2 3
    def get_range_grid_index(self):
        gap = self.grid_num[0]/self.size
        self.start_x = self.global_rank*gap
        self.end_x = self.start_x + calc_ncols_from_rank(self.global_rank ,self.size, self.grid_size[0])
        self.start_grid_index= self.start_x*self.grid_num[1]*self.grid_num[2]
        self.end_grid_index = self.end_x*self.grid_num[1]*self.grid_num[2] - 1
        if self.gpu_node_rank > 0:
            self.all_start_grid_index = self.start_x - self.slice_size
        else:
            self.all_start_grid_index = self.start_x 
        if self.gpu_node_rank < self.gpu_node_size:
            self.all_end_grid_index = self.end_x + self.slice_size
        else:
            self.all_end_grid_index = self.end_x
            
    def get_border_grid_range(self):
        self.left_border_grid_start_index = self.start_grid_index 
        self.left_border_grid_end_index = self.start_grid_index + self.slice_size -1
        
        self.right_border_grid_start_index = self.end_grid_index - self.slice_size + 1
        self.right_border_grid_end_index = self.end_grid_index
        
    def get_ghost_grid_range(self):
        self.left_ghost_grid_start_index = self.start_grid_index - self.slice_size
        self.left_ghost_grid_end_index = self.start_grid_index - 1
        
        self.right_ghost_grid_start_index = self.end_grid_index + 1
        self.right_ghost_grid_end_index = self.end_grid_index + self.slice_size
        
    def split(self):
        req = self.comm.irecv(0, 1)
        self.local_particle_num = req.wait()
        if self.gpu_node_rank > 0:
            reqleft = self.comm.irecv(0, 11)
            self.left_ghost_num = reqleft.wait()
        else:
            self.left_ghost_num = 0
        if self.gpu_node_rank < self.gpu_node_rank:
            reqright = self.comm.irecv(0, 22)
            self.right_ghost_num = reqright.wait()
        else:
            self.right_ghost_num = 0
        
        # 将幽灵粒子纳入到统一的体系中，但是在更新的时候不对其进行更新
        self.particle_num = self.local_particle_num + self.left_ghost_num + self.right_ghost_num
        self.grid_ids = ti.field(dtype=ti.i32, shape=self.particle_num)
        self.x = ti.Vector.field(self.dim, ti.f32, shape=self.particle_num)
        self.v = ti.Vector.field(self.dim, ti.f32, shape=self.particle_num)
        self.density = ti.field(dtype=ti.f32, shape=self.particle_num)
        self.pressure = ti.field(dtype=ti.f32, shape=self.particle_num)
        self.material = ti.field(dtype=ti.i32, shape=self.particle_num)
        self.color = ti.field(dtype=ti.i32, shape=self.particle_num)
        self.mass = ti.field(dtype=ti.f32, shape=self.particle_num) # 单个粒子的质量
        self.volume = ti.field(dtype=ti.f32, shape=self.particle_num)
        self.paritcle_index_temp = ti.field(dtype=ti.i32, shape=self.particle_num)
        self.paritcle_index = ti.field(dtype=ti.i32, shape=self.particle_num)
        self.grid_particles_num = ti.field(dtype=ti.i32, shape=self.particle_num)

        # 进行排序时的buffer
        self.grid_ids_buffer = ti.field(dtype=ti.i32, shape=self.particle_num)
        self.x_buffer = ti.Vector.field(self.dim, ti.f32, shape=self.particle_num)
        self.v_buffer = ti.Vector.field(self.dim, ti.f32, shape=self.particle_num)
        self.density_buffer = ti.field(dtype=ti.f32, shape=self.particle_num)
        self.pressure_buffer = ti.field(dtype=ti.f32, shape=self.particle_num)
        self.material_buffer = ti.field(dtype=ti.i32, shape=self.particle_num)
        self.color_buffer = ti.field(dtype=ti.i32, shape=self.particle_num)
        self.mass_buffer = ti.field(dtype=ti.f32, shape=self.particle_num) # 单个粒子的质量
        self.volume_buffer = ti.field(dtype=ti.f32, shape=self.particle_num)

        req1 = self.comm.irecv(0, 2)
        x = req1.wait()
        
        req2 = self.comm.irecv(0, 3)
        v = req2.wait()
        
        req3 = self.comm.irecv(0, 4)
        density = req3.wait()
        
        req4 = self.comm.irecv(0, 5)
        pressure = req4.wait()
        
        req5 = self.comm.irecv(0, 6)
        material = req5.wait()
        
        req6 = self.comm.irecv(0, 7)
        color = req6.wait()
        
        req7 = self.comm.irecv(0, 8)
        mass = req7.wait()
        
        req8 = self.comm.irecv(0, 9)
        volume = req8.wait()
        # 区域内粒子的属性信息初始化
        self.x.from_numpy(x)
        self.density.from_numpy(density)
        self.v.from_numpy(v)
        self.pressure.from_numpy(pressure)
        self.color.from_numpy(color)
        self.material.from_numpy(material)
        self.mass.from_numpy(mass)
        self.volume.from_numpy(volume)
        # 获取到各个区域的网格id信息
        self.get_ghost_grid_range()
        self.get_border_grid_range()
        self.get_range_grid_index()
        
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.resort()

    @ti.kernel
    def update_grid_id(self):
        # 更新内部粒子的网格id
        for i in ti.grouped(self.local_grid_particle_num):
            self.local_grid_particle_num[i] = 0
        for i in ti.grouped(self.x):
            grid_index = self.get_flatten_grid_index(self.x[i])
            local_grid_index = grid_index - self.all_start_grid_index
            self.grid_ids[i] = grid_index
            ti.atomic_add(self.local_grid_particle_num[local_grid_index], 1)
        for i in ti.grouped(self.local_grid_particle_num):
            self.local_grid_particle_num_temp[i] = self.local_grid_num[i]
    
    @ti.kernel
    def exchange_particles(self):
        pass
    
    @ti.kernel
    def resort(self):
        # 对区域内粒子进行排序
        for i in range(self.particle_num):
            j = self.particle_max_num - 1 -i
            offset = 0
            # 得到本地的网格id
            local_grid_id = self.grid_ids[j] - self.all_start_grid_index
            if local_grid_id - 1 >= 0:
                offset = self.grid_particles_num[local_grid_id - 1]
            self.paritcle_index_temp[j] = ti.atomic_sub(self.grid_particles_num_temp[local_grid_id], 1) - 1 + offset

        for i in ti.grouped(self.grid_ids):
            newIndex = self.paritcle_index_temp[i]
            self.grid_ids_buffer[newIndex] = self.grid_ids[i]
            self.m_buffer[newIndex] = self.m[i]
            self.v_buffer[newIndex] = self.v[i]
            self.x_buffer[newIndex] = self.x[i]
            self.density_buffer[newIndex] = self.density[i]
            self.pressure_buffer[newIndex] = self.pressure[i]
            self.material_buffer[newIndex] = self.material[i]
            self.color_buffer[newIndex] = self.color[i]
            self.mass_buffer[newIndex] = self.mass[i]
            self.volume_buffer[newIndex] = self.volume[i]
            
        for i in ti.grouped(self.x):
            self.grid_ids[i] = self.grid_ids_buffer[i]
            self.m[i] = self.m_buffer[i]
            self.v[i] = self.v_buffer[i]
            self.x[i] = self.x_buffer[i]
            self.density[i] = self.density_buffer[i]
            self.pressure[i] = self.pressure_buffer[i]
            self.material[i] = self.material_buffer[i]
            self.color[i] = self.color_buffer[i]
            self.mass[i] = self.mass_buffer[i]
            self.volume[i] = self.volume_buffer[i]

    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        """visit the particle p_i and 

        Args:
            p_i (_type_): _description_
            task (ti.template): _description_
            ret (ti.template): _description_
        """
        center_cell = self.pos_to_index(self.x[p_i]) - self.all_start_grid_index
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
            # 获取到局域内部的index
            grid_index = self.flatten_grid_index(center_cell + offset) 
            # 不能超过ghost的区域
            for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[ti.max(grid_index, self.local_grid_count)]):
                if p_i != p_j and (self.x[p_i] - self.x[p_j]).norm() < self.support_length:
                    task(p_i, p_j, ret)
    
    def update(self):
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.resort()
