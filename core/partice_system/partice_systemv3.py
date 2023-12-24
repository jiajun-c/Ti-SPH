import taichi as ti
import numpy as np
import trimesh as trim
from functools import reduce


@ti.data_oriented
class ParticleSystemV3:
    def __init__(self, res, simulation_config):
        self.res = res
        self.dim = len(res)
        assert self.dim > 1
        self.screen_to_world_ratio = 50
        self.bound = np.array(res) / self.screen_to_world_ratio
        print(self.bound)
        # 材料相关性质
        self.material_boundary = 0
        self.material_fluid = 1
        self.simulation_config = simulation_config
        # # 粒子相关信息
        self.particle_radius = 0.05  # particle radius
        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V = 0.8 * self.particle_diameter ** self.dim # 粒子的体积
        self.particle_max_num = 2 ** 15
        self.particle_max_num_per_cell = 100
        self.particle_max_num_neighbor = 100
        self.particle_num = ti.field(int, shape=())
        # # 网格相关属性
        self.grid_size = self.support_radius
        self.grid_num = np.ceil(np.array(res) / self.grid_size).astype(int)
        print("grid_name", self.grid_num)
        self.grid_particles_num = ti.field(int)
        self.grid_particles = ti.field(int)
        self.padding = self.grid_size
        # 模拟的配置
        self.config = self.simulation_config['configuration']
        
        # # 固体
        self.rigidBodiesConfig = self.simulation_config['rigidBodies']  # list
        self.fluidBlocksConfig = self.simulation_config['fluidBlocks']  # list

        # # Snode数据结构
        self.x = ti.Vector.field(self.dim, dtype=float)
        self.v = ti.Vector.field(self.dim, dtype=float)
        self.density = ti.field(dtype=float)
        self.pressure = ti.field(dtype=float)
        self.material = ti.field(dtype=int)
        self.color = ti.field(dtype=int)
        self.particle_neighbors = ti.field(int)
        self.particle_neighbors_num = ti.field(int)
        self.volume = ti.field(dtype=ti.f32)

        self.particles_node = ti.root.dense(ti.i, self.particle_max_num)
        self.particles_node.place(self.x, self.v, self.density, self.pressure, self.material, self.color, self.volume)
        self.particles_node.place(self.particle_neighbors_num)
        self.particle_node = self.particles_node.dense(ti.j, self.particle_max_num_neighbor)
        self.particle_node.place(self.particle_neighbors)   

        index = ti.ij if self.dim == 2 else ti.ijk
        grid_node = ti.root.dense(index, self.grid_num)
        grid_node.place(self.grid_particles_num)

        cell_index = ti.k if self.dim == 2 else ti.l
        cell_node = grid_node.dense(cell_index, self.particle_max_num_per_cell)
        cell_node.place(self.grid_particles)

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
        
    def add_fluid_and_rigid(self):
        """add fluid and rigid body declared in the configuration file
        """
        
        # add rigid body
        for rigid in self.rigidBodiesConfig:
            voxelized_points = self.load_rigid_body(rigid)
            particle_num = voxelized_points.shape[0]
            # print(self.particle_num[None])
            # self.particle_num[None] += particle_num
            rigid['partice_num'] = particle_num
            rigid['voxelized_points'] = voxelized_points
            # TODO: add the material type for the rigid body instead of the boundary material
            material = np.full((particle_num, ), self.material_boundary, dtype=np.int32)
            color = rigid['color']
            if type(color[0]) == int:
                color = [c / 255.0 for c in color]
            color  = np.tile(np.array(color, dtype=np.float32), (particle_num, 1))
            velocity = rigid['velocity']
            velocity = np.tile(np.array(velocity, dtype=np.float32), (particle_num, 1))
            
            density = rigid['density']
            # print(particle_num)
            density = np.full_like(np.zeros(particle_num), density if density is not None else 1000.)

            pressure = np.full_like(np.zeros(particle_num), 0.)

            positions = voxelized_points
            self.add_particles(particle_num, 
                            positions,
                            velocity,
                            density,
                            pressure,
                            material,
                            color)
            
        # add fluid blocks
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
        
    @ti.func
    def add_particle(self, p, x, v, density, pressure, material, color):
        self.x[p] = x
        self.v[p] = v
        self.density[p] = density
        self.pressure[p] = pressure
        self.material[p] = material
        self.color[p] = color

    @ti.kernel
    def add_particles(self, num: int,
                    particle_position: ti.types.ndarray(),
                    particle_velocity: ti.types.ndarray(),
                    particle_density: ti.types.ndarray(),
                    particle_pressure: ti.types.ndarray(),
                    particle_material: ti.types.ndarray(),
                    particle_color: ti.types.ndarray()):
        for i in range(num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            color = ti.Vector.zero(float, self.dim)
            for j in ti.static(range(self.dim)):
                v[j] = particle_velocity[i, j]
                x[j] = particle_position[i, j]
            #     color[j] = particle_color[i, j]
            self.add_particle(self.particle_num[None] + i, x, v,
                            particle_density[i],
                            particle_pressure[i],
                            particle_material[i],
                            particle_color[i])
        self.particle_num[None] += num

    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)

    @ti.func
    def is_valid_cell(self, cell):
        flag = True
        for i in ti.static(range(self.dim)):
            flag = flag and (0 <= cell[i] < self.grid_num[i])
        return flag

    @ti.kernel
    def search_neighbors(self):
        for p_i in range(self.particle_num[None]):
            if self.material[p_i] == self.material_boundary:
                continue
            center_cell = self.pos_to_index(self.x[p_i])
            cnt = 0
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 2)):
                cell = center_cell + offset
                if not self.is_valid_cell(cell):
                    break
                for j in range(self.grid_particles_num[cell]):
                    p_j = self.grid_particles[cell, j]
                    if p_j == p_i:
                        continue
                    if (self.x[p_i] - self.x[p_j]).norm() >= self.support_radius:
                        continue
                    self.particle_neighbors[p_i, cnt] = p_j
                    cnt += 1
            self.particle_neighbors_num[p_i] = cnt
    @ti.func
    def for_all_neighbors(self, idx_i, task: ti.template(), ret: ti.template()):
        for idx_j in range(self.particle_neighbors_num[idx_i]):
            task(idx_i, idx_j, ret)
    
    @ti.kernel
    def allocate_particles_to_grid(self):
        for i in range(self.particle_num[None]):
            cell = self.pos_to_index(self.x[i])
            offset = ti.atomic_add(self.grid_particles_num[cell], 1)
            self.grid_particles[cell, offset] = i

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
                            self.particle_radius))
        num_new_particles = reduce(lambda x, y: x * y,
                                    [len(n) for n in num_dim])
        # assert self.particle_num[None] + num_new_particles <= self.particle_max_num

        positions = np.array(np.meshgrid(*num_dim,
                                        sparse=False,
                                        indexing='ij'),
                            dtype=np.float32)
        positions = positions.reshape(-1,
                                      reduce(lambda x, y: x * y, list(positions.shape[1:]))).transpose()
        velocity = np.full(positions.shape, fill_value=0 if velocity is None else velocity, dtype=np.float32)
        material = np.full_like(np.zeros(num_new_particles), material)
        color = np.full_like(np.zeros(num_new_particles), color)
        density = np.full_like(np.zeros(num_new_particles), density if density is not None else 1000.)
        pressure = np.full_like(np.zeros(num_new_particles), pressure if pressure is not None else 0.)
        print("add")
        self.add_particles(num_new_particles, positions, velocity, density, pressure, material, color)

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]

    @ti.kernel
    def copy_to_numpy_nd(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            for j in ti.static(range(self.dim)):
                np_arr[i, j] = src_arr[i][j]

    def compute_fluid_particle_num(self, start, end):
        """computer the fluid particle number

        Args:
            start (_type_): the start point
            end (_type_): the end point

        Returns:
            _type_:the particle number
        """
        particle_num = 1
        for i in range(self.dim):
            particle_num *= len(np.arange(start[i], end[i], self.particle_diameter))
        return particle_num
    
    def dump(self):
        print(self.particle_num[None])
        np_x = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(np_x, self.x)

        np_v = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(np_v, self.v)

        np_material = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_material, self.material)

        np_color = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_color, self.color)
        return {
            'position': np_x,
            'velocity': np_v,
            'material': np_material,
            'color': np_color
        }

    def init(self):
        print("init")
        self.grid_particles_num.fill(0)
        self.particle_neighbors.fill(0)
        self.allocate_particles_to_grid()
        self.search_neighbors()
