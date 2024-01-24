import taichi as ti
import numpy as np
from functools import reduce

@ti.data_oriented
class ParticleBase:
    def __init__(self, simulation_config):
        self.simulation_config = simulation_config
        self.configuration = self.simulation_config['configuration']
        self.rigidBodiesConfig = self.simulation_config['rigidBodies']  # rigid 
        self.fluidBlocksConfig = self.simulation_config['fluidBlocks']  # fluid
        
        self.dim = self.configuration['dim']
        
        self.domain_start = np.array(self.configuration['domainStart'])
        self.domain_end = np.array(self.configuration['domainEnd'])
        self.domain_size = self.domain_end - self.domain_start
        
        # particles info
        self.particle_radius = self.configuration['particleRadius']
        self.particle_diameter = 2 * self.particle_radius
        self.support_length = 4.0 * self.particle_radius
        self.padding = self.support_length  # padding is used for boundary condition when particle collide with wall
        self.particle_num = ti.field(int, shape=())
        self.particle_max_num = 0
        self.compute_particle_num() # sum of the particles
        self.m = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)
        self.volume = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.x = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num) # the particle position
        self.density = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.pressure = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.material = ti.field(dtype=ti.i32, shape=self.particle_max_num)
        self.object_id = ti.field(dtype=ti.int32, shape=self.particle_max_num)
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim # 粒子的体积
        self.mass = ti.field(dtype=ti.f32, shape=self.particle_max_num) #
        self.d_velocity = ti.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)
        
        self.add_fluid_and_rigid() # compute the particle_max_num
        self.grid_size = self.support_length
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(np.int32)
        self.grid_particles_num = ti.field(int, shape=int(reduce(lambda x, y : x*y, self.grid_num)))
        self.grid_particles_num_temp = ti.field(int, shape=int(reduce(lambda x, y : x*y, self.grid_num)))
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])
        self.grid_ids = ti.field(int, shape=self.particle_max_num)
        self.paritcle_index_temp = ti.field(int, shape=self.particle_max_num) # paritcle position after resort 
        
        # the buffer for sort
        self.grid_ids_buffer = ti.field(int, shape=self.particle_max_num)
        self.m_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.v_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num) # the particle position
        self.density_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.material_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.object_id_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.volume_buffer = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.mass_buffer = ti.field(dtype=ti.f32,shape=self.particle_max_num)
    def compute_particle_num(self):
        pass
    
    def add_fluid_and_rigid(self):
        pass
    
    @ti.func
    def is_valid_cell(self, cell):
        """

        Args:
            cell (grid index of the particle)

        Returns:
            bool: valid return true
        """
        flag = True
        for i in ti.static(range(self.dim)):
            flag = flag and (0 <= cell[i] < self.grid_num[i])
        return flag

    @ti.func
    def pos_to_index(self, pos):
        """turn particle position into grid index

        Args:
            pos (vec): position

        Returns:
            index: grid index
        """
        assert pos[1] < self.domain_size[1] and pos[2] < self.domain_size[2] and pos[0] < self.domain_size[0]
        ans = (pos / self.grid_size).cast(int)
        if pos[0] >= self.domain_size[0] or pos[1] >= self.domain_size[1] or pos[2] >= self.domain_size[2]:
            print(ans)
        return (pos / self.grid_size).cast(int)
    
    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))
    
    @ti.func
    def flatten_grid_index(self, grid_index):
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]
    
    def add_fluid_and_rigid(self):
        pass
    
    @ti.func
    def add_particle(self, p, x, v, density, pressure, material, id):
        # if p >= self.particle_max_num:
            # print("error:", self.particle_num[None])
        self.x[p] = x
        self.v[p] = v
        self.density[p] = density
        self.pressure[p] = pressure
        self.material[p] = material
        self.object_id[p] = id
        self.volume[p] = self.m_V0 # TODO: fluid volume need be compute later
        self.mass[p] = self.volume[p]*self.density[p]

    @ti.kernel
    def add_particles(self, num: int,
                    particle_position: ti.types.ndarray(),
                    particle_velocity: ti.types.ndarray(),
                    particle_density: ti.types.ndarray(),
                    particle_pressure: ti.types.ndarray(),
                    particle_material: ti.types.ndarray(),
                    object_id: int):
        print("add now", self.particle_num[None])
        for i in range(num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for j in ti.static(range(self.dim)):
                v[j] = particle_velocity[i, j]
                x[j] = particle_position[i, j]
            self.add_particle(self.particle_num[None] + i, x, v,
                            particle_density[i],
                            particle_pressure[i],
                            particle_material[i],
                            object_id)
        self.particle_num[None] += num
        
    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        """visit the particle p_i and 

        Args:
            p_i (_type_): _description_
            task (ti.template): _description_
            ret (ti.template): _description_
        """
        center_cell = self.pos_to_index(self.x[p_i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
            grid_index = self.flatten_grid_index(center_cell + offset)
            for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[grid_index]):
                if p_i != p_j and (self.x[p_i] - self.x[p_j]).norm() < self.support_length:
                    task(p_i, p_j, ret)