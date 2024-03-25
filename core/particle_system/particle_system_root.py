import taichi as ti
import numpy as np
import trimesh as trim
from functools import reduce
from paritcle_system_base import ParticleSystemBase
from mpi4py import MPI

def calc_ncols_from_rank(rank, size, NCOLS):
    ncols = int(NCOLS / size)       
    if ((NCOLS % size) != 0):
        if (rank == size - 1):
            ncols += NCOLS % size
    return ncols

@ti.data_oriented
class ParticleSystemRoot(ParticleSystemBase):
    def __init__(self, simulation_config) -> None:
        super.__init__(simulation_config)
        # 最佳情况网格应该为128, 128, 128这样
        self.particle_max_num = 0
        self.particle_num = ti.field(int, shape=())
        self.compute_particle_num() # sum of the particles
        self.comm = MPI.COMM_WORLD
        self.gpu_node_size = self.comm.size - 1
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim # 粒子的体积
        self.material_boundary = 0
        self.material_fluid = 1
        # 粒子的属性信息，如质量，
        self.m = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)
        self.volume = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.pressure = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.material = ti.field(dtype=ti.i32, shape=self.particle_max_num)
        self.color = ti.Vector.field(3, dtype=ti.int32, shape=self.particle_max_num)
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim # 粒子的体积
        self.mass = ti.field(dtype=ti.f32, shape=self.particle_max_num) # 单个粒子的质量

        self.grid_particles_num = ti.field(int, shape=int(reduce(lambda x, y : x*y, self.grid_num)))
        self.grid_particles_num_temp = ti.field(int, shape=int(reduce(lambda x, y : x*y, self.grid_num)))
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

        self.x = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num) # the particle position
        self.density = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.add_fluid_and_rigid() # compute the particle_max_num
    def split(self):
        pass
    
    def add_fluid_and_rigid(self):
        for rigid in self.rigidBodiesConfig:
            voxelized_points = self.load_rigid_body(rigid)
            velocity = rigid['velocity']
            density = rigid['density']
            self.add_particles(voxelized_points, velocity, density, 0, self.material_boundary)
            
        for fluid in self.fluidBlocksConfig:
            start = fluid['start']
            end = fluid['end']
            velocity = fluid['velocity']
            color = fluid['color']
            density = fluid['density']
            cube_size = [end[0] - start[0], end[1] - start[1], end[2] - start[2]]
            self.add_cube(lower_corner=start, 
                        cube_size=cube_size, 
                        material= self.material_fluid,
                        color=0x111111,
                        density=density,
                        velocity=velocity)

    def load_rigid_body(self, rigid_body):
        """load the rigid body

        Args:
            rigid_body (json): the configuration of the rigid body
        """
        # the mesh only contains the surface information
        mesh = trim.load(rigid_body['geometryFile'])
        mesh.apply_scale(rigid_body['scale'])
        offset = np.array(rigid_body['translation'])
        rotation_angle = rigid_body['rotationAngle'] * np.pi / 180
        rotation_axis = rigid_body['rotationAxis']
        rot_matrix = trim.transformations.rotation_matrix(rotation_angle, rotation_axis, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)
        mesh.vertices += offset
        rigid_body['mesh'] = mesh.copy()
        # self.get_mesh_info(mesh)
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).fill()
        return voxelized_mesh.points.astype(np.float32)

    @ti.kernel
    def add_particles(self, num: int,
                    particle_position: ti.types.ndarray(),
                    particle_velocity: ti.types.ndarray(),
                    particle_density: ti.types.ndarray(),
                    particle_pressure: ti.types.ndarray(),
                    particle_material: ti.types.ndarray(),
                    particle_color: ti.types.ndarray()):
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
                            particle_color[i])
        self.particle_num[None] += num
        
    def add_cube(self,
                lower_corner,
                cube_size,
                material,
                color=0xFFFFFF,
                density=None,
                pressure=None,
                velocity=None):

        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                            self.particle_diameter))
        num_new_particles = reduce(lambda x, y: x * y,
                                    [len(n) for n in num_dim])

        # assert self.particle_num[None] + num_new_particles <= self.particle_max_num
        positions = np.array(np.meshgrid(*num_dim, indexing='ij'), dtype=np.float32)
        positions = positions.reshape(self.dim, num_new_particles).T
        velocity = np.full(positions.shape, fill_value=0 if velocity is None else velocity, dtype=np.float32)
        material = np.full_like(np.zeros(num_new_particles), material)
        color = np.full_like(np.zeros(num_new_particles), color)
        density = np.full_like(np.zeros(num_new_particles), density if density is not None else 1000.)
        pressure = np.full_like(np.zeros(num_new_particles), pressure if pressure is not None else 0.)
        print("shape", positions.shape)
        self.add_particles(num_new_particles, positions, velocity, density, pressure, material, color)



    def compute_particle_num(self):
        for fluid in self.fluidBlocksConfig:
            start = fluid['start']
            end = fluid['end']
            self.particle_max_num += self.compute_cube_particles_num(start, end)
            # print(self.particle_max_num)
        for rigid in self.rigidBodiesConfig:
            voxelized_points = self.load_rigid_body(rigid)
            particle_num = voxelized_points.shape[0]
            self.particle_max_num += particle_num
            
    def split_particles(self):
        ranges = []
        start_grid_index = 0
        # for i in range(self.comm.size):
        #     sub_grid_size = calc_ncols_from_rank(i, self.comm.size, self.grid_num[0])*self.grid_num[1]*self.grid_num[2]
        #     ranges = ranges.append(start_grid_index)
        #     start_grid_index = start_grid_index + sub_grid_size
        for i in range(self.particle_num):
            x = self.x[0]
            gap = int(self.grid_num[0]/self.comm.size)
            belong = int(x/gap)
            if belong >= self.gpu_node_size:
                belong -= 1

    @ti.kernel
    def resort(self):
        for i in range(self.particle_max_num):
            j = self.particle_max_num - 1 -i
            offset = 0
            if self.grid_ids[j] - 1 >= 0:
                offset = self.grid_particles_num[self.grid_ids[j] - 1]
            self.paritcle_index_temp[j] = ti.atomic_sub(self.grid_particles_num_temp[self.grid_ids[j]], 1) - 1 + offset

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
            
    @ti.kernel
    def update_gird_id(self):
        for i in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[i] = 0
        for I in ti.grouped(self.x):
            grid_index = self.get_flatten_grid_index(self.x[I])
            self.grid_ids[I] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)
        for i in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[i] = self.grid_particles_num[i]
            
    def split(self):
        self.resort()
        self.update_gird_id()
        start_grid_id = 0
        x_np = self.x.to_numpy()
        v_np = self.v.to_numpy()
        density_np = self.density.to_numpy()
        pressure_np = self.pressure.to_numpy()
        material_np = self.material.to_numpy()
        color_np = self.material.to_numpy()
        mass_np = self.mass.to_numpy()
        volume_np = self.volume.to_numpy()
        for i in range(self.gpu_node_size):
            node_grid_size = calc_ncols_from_rank(i, self.gpu_node_size, self.grid_num[0])
            if i == 0:
                node_particle_size = self.grid_particles_num[start_grid_id + node_grid_size-1]
            else:
                node_particle_size = self.grid_particles_num[start_grid_id + node_grid_size-1] - self.grid_particles_num[start_grid_id-1]
            
            self.comm.isend(node_particle_size, i+1, 1)
            if i > 0:
                left_ghost_particle_size = self.grid_particles_num[start_grid_id-1] - self.grid_particles_num[start_grid_id-self.slice_size-1]
                self.comm.isend(left_ghost_particle_size, i+1, 11)
            else:
                left_ghost_particle_size = 0
            if i < self.gpu_node_size:
                right_ghost_particle_size = self.grid_particles_num[start_grid_id+self.slice_size-1] - self.grid_particles_num[start_grid_id-1]
                self.comm.isend(right_ghost_particle_size, i+1, 22)
            else:
                right_ghost_particle_size = 0 
            l = start_grid_id*self.slice_size - left_ghost_particle_size
            r = start_grid_id*self.slice_size + node_particle_size + right_ghost_particle_size
            self.comm.isend(x_np[l:r], i+1, 2)
            self.comm.isend(v_np[l:r], i+1, 3)
            self.comm.isend(density_np[l:r], i+1, 4)
            self.comm.isend(pressure_np[l:r], i+1, 5)
            self.comm.isend(material_np[l:r], i+1, 6)
            self.comm.isend(color_np[l:r], i+1, 7)
            self.comm.isend(mass_np[l:r], i+1, 8)
            self.comm.isend(volume_np[l:r], i+1, 9)
            start_grid_id += node_grid_size
            
            