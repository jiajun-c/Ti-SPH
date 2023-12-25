import taichi as ti
import trimesh as trim
import numpy as np
from functools import reduce

@ti.data_oriented
class ParticleSystemV2:
    def __init__(self, simulation_config):
        assert self.dim > 1
        
        # The configuration 
        self.simulation_config = simulation_config
        self.configuration = self.simulation_config['configuration']
        
        # The domin scope
        self.domain_start = self.configuration['domainStart']
        self.domain_end = self.configuration['domainEnd']
        self.dim = len(self.domin_end)
        self.domain_size = self.domain_end - self.domain_start
        
        self.material_boundary = 0
        self.material_fluid = 1
        
        self.fluid = self.simulation_config['fluidBlocks']
        self.rigid = self.simulation_config['rigidBodies']
        
        self.padding = self.support_length  # padding is used for boundary condition when particle collide with wall
        self.grid_size = self.support_radius
        
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(np.int32)
        
        # particles info
        self.particle_radius = self.configuration['particleRadius']
        self.support_length = 4 * self.particle_radius
        self.particle_max_num = 0 # sum of the particles
        self.add_fluid_and_rigid() # compute the particle_max_num
        self.m = ti.field(dtype=float, shape=self.particle_max_num)
        self.v = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.density = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure = ti.field(dtype=float, shape=self.particle_max_num)
        self.material = ti.field(dtype=int, shape=self.particle_max_num)
        self.color = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim # 粒子的体积
        
        # grid info
        """after substep,we need to resort the particles.
            | gird(0, 0, 0) nodes | grid(0, 0, 1) nodes |...
        """
        index = ti.ij if self.dim == 2 else ti.ijk
        self.grid_particles_num = ti.field(int, shape=int(reduce(lambda x, y : x*y, self.grid_num)))
        
        
        # the buffer for sort
        
    def add_fluid_and_rigid(self):
        pass
    
    def compute_cube_particles_num(self, start, end):
        pass
    
    def compute_body_particles_num(self, start, end):
        pass
    
    def add_particles(self):
        pass
    
    def update_gird_id(self):
        for i in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[i] = 0
        
    def update(self):
        """After sph substep, the particles system should be updated
        """
        
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