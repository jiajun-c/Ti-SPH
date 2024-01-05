import taichi as ti
import trimesh as trim
import numpy as np
from functools import reduce

@ti.data_oriented
class ParticleSystemV4:
    def __init__(self, simulation_config):
        
        # The configuration 
        self.simulation_config = simulation_config
        self.configuration = self.simulation_config['configuration']
        self.rigidBodiesConfig = self.simulation_config['rigidBodies']  # list
        self.fluidBlocksConfig = self.simulation_config['fluidBlocks']  # list
        self.density0 = self.configuration['density0']

        self.dim = self.configuration['dim']
        # The domin scope
        
        self.domain_start = np.array(self.configuration['domainStart'])
        self.domain_end = np.array(self.configuration['domainEnd'])
        self.domain_size = self.domain_end - self.domain_start
        
        self.material_boundary = 0
        self.material_fluid = 1
        self.material_rigid_dynamic = 2
        
        self.fluid = self.simulation_config['fluidBlocks']
        self.rigid = self.simulation_config['rigidBodies']
        
        
        
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
        self.mass = ti.field(dtype=ti.f32, shape=self.particle_max_num) # 单个粒子的质量
        # self.particles_node = ti.root.dense(ti.i, self.particle_max_num)
        # self.particles_node.place(self.x)
        self.add_fluid_and_rigid() # compute the particle_max_num

        # grid info
        """after substep,we need to resort the particles.
            | gird(0, 0, 0) nodes | grid(0, 0, 1) nodes |...
        """
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
    @ti.func
    def is_valid_cell(self, cell):
        flag = True
        for i in ti.static(range(self.dim)):
            flag = flag and (0 <= cell[i] < self.grid_num[i])
        return flag

    @ti.func
    def pos_to_index(self, pos):
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

        for rigid in self.rigidBodiesConfig:
            voxelized_points = self.load_rigid_body(rigid)
            particle_num = voxelized_points.shape[0]
            # print(self.particle_num[None])
            rigid['partice_num'] = particle_num
            rigid['voxelized_points'] = voxelized_points
            material = self.material_boundary
            """
            If a body is dynamic, it can be 
            """
            if rigid['dynamic']:
                material = self.material_rigid_dynamic
            material = np.full((particle_num, ), material, dtype=np.int32)
            # color = rigid['color']
            # if type(color[0]) == int:
            #     color = [c / 255.0 for c in color]
            # color  = np.tile(np.array(color, dtype=np.float32), (particle_num, 1))
            velocity = rigid['velocity']
            velocity = np.tile(np.array(velocity, dtype=np.float32), (particle_num, 1))
            
            density = rigid['density']
            # print(particle_num)
            density = np.full_like(np.zeros(particle_num), density if density is not None else 1000.)

            pressure = np.full_like(np.zeros(particle_num), 0.)
            color = np.full_like(np.zeros(particle_num), 0xFFFFFF)

            positions = voxelized_points
            self.add_particles(particle_num, 
                            positions,
                            velocity,
                            density,
                            pressure,
                            material,
                            1)
            # self.particle_num[None] += voxelized_points.shape[0]
            
        for fluid in self.fluidBlocksConfig:
            start = fluid['start']
            end = fluid['end']
            velocity = fluid['velocity']
            color = fluid['color']
            density = fluid['density']
            cube_size = [end[0] - start[0], end[1] - start[1], end[2] - start[2]]
            object_id = fluid['objectId']
            self.add_cube(lower_corner=start, 
                        cube_size=cube_size, 
                        material= self.material_fluid,
                        color=0x111111,
                        density=density,
                        velocity=velocity,
                        object_id=object_id)

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


    def compute_cube_particles_num(self, start, end):
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(start[i], end[i],
                            self.particle_diameter))
        num_new_particles = reduce(lambda x, y: x * y,
                                    [len(n) for n in num_dim])
        return num_new_particles
        
        
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
    def update_gird_id(self):
        for i in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[i] = 0
        for I in ti.grouped(self.x):
            grid_index = self.get_flatten_grid_index(self.x[I])
            self.grid_ids[I] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)
        for i in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[i] = self.grid_particles_num[i]
    
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
            self.object_id_buffer[newIndex] = self.object_id[i]
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
            self.object_id[i] = self.object_id_buffer[i]
            self.mass[i] = self.mass_buffer[i]
            self.volume[i] = self.volume_buffer[i]
            
    def update(self):
        """After sph substep, the particles system should be updated
        """
        self.update_gird_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.resort()
        
        pass
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
    
    def dump(self):
        np_x = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(np_x, self.x)

        np_v = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(np_v, self.v)

        np_material = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_material, self.material)

        np_object_id = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_object_id, self.object_id)
        return {
            'position': np_x,
            'velocity': np_v,
            'material': np_material,
            'objectID': np_object_id
        }
        
    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]

    @ti.kernel
    def copy_to_numpy_nd(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            for j in ti.static(range(self.dim)):
                np_arr[i, j] = src_arr[i][j]

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

    def add_cube(self,
                lower_corner,
                cube_size,
                material,
                color=0xFFFFFF,
                density=None,
                pressure=None,
                velocity=None,
                object_id=-1):

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
        self.add_particles(num_new_particles, positions, velocity, density, pressure, material, object_id)

    @ti.func
    def is_dynamic(self, material):
        return material == self.material_fluid or material == self.material_rigid_dynamic
