import taichi as ti
import trimesh as trim
import numpy as np
from scipy.spatial.transform import Rotation

ti.init(ti.cuda, debug=True)
matrix_type = ti.Matrix([[0] * 5 for _ in range(5)], ti.f32)
particles_pos = ti.Vector.field(3, dtype=ti.f32, shape = 3)
@ti.kernel
def init_points_pos(points : ti.template()):
    for i in range(points.shape[0]):
        points[i] = [i for j in ti.static(range(3))]

@ti.data_oriented
class Bunny:
    def __init__(self, filepath):
        self.rigid = trim.load(filepath)
        mesh = trim.load(filepath)
        # self.get_mesh_info(mesh)
        vertices = mesh.vertices
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=vertices.shape[0])
        for i in range(vertices.shape[0]):
            vertices[i] = vertices[i] * 10
        self.vertices = ti.Vector.field(3, dtype=float, shape=vertices.shape[0])
        self.vertices.from_numpy(mesh.vertices)
        # 转动惯量
        self.I_ref = ti.Matrix([[0.0] * 3 for _ in range(3)], ti.f32)
        self.I_ref[0, 0] = 1.0
        for i in range(0, self.vertices.shape[0]):
            x = self.vertices[i][0]
            y = self.vertices[i][1]
            z = self.vertices[i][2]
            self.I_ref[0,0] += x*x + y*y + z*z
            self.I_ref[1,1] += x*x + y*y + z*z
            self.I_ref[2,2] += x*x + y*y + z*z
            self.I_ref[0,0] -= x*x
            self.I_ref[0,1] -= x*y
            self.I_ref[0,2] -= x*z
            
            self.I_ref[1,0] -= y*x
            self.I_ref[1,1] -= y*y
            self.I_ref[1,2] -= y*z
            
            self.I_ref[2,0] -= z*x
            self.I_ref[2,1] -= z*y
            self.I_ref[2,2] -= z*z
        self.q = ti.Vector.field(4, dtype=ti.f32, shape=())
        self.q[None][0] = 1.0
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.w        = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.w[None][0] = 0
        self.w[None][1] = 2
        self.w[None][2] = 0
        self.velocity[None][0] = 0.1
        self.velocity[None][1] = 0.1
        self.velocity[None][2] = 0
        for i in range(3):
            self.position[None][i] = 0.1
            self.velocity[None][i] = 0.0
            self.w[None][i] = 0.0
        self.sumv = ti.Vector([0.0, 0.0, 0.0])
        self.linear_decay = 0.999
        self.g = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.g[None][1] = -9.8
        self.dt = 0.003
        self.times = 10

    # @ti.kernel
    def calulate_Iref(self):
        self.I_ref =  self.calulate_Iref_func()
        
    @ti.kernel
    def calulate_Iref_func(self) -> ti.Matrix:
        I_ref = ti.Matrix([[0]* 3 for _ in  range(3)], ti.f32)

        return I_ref
            # pass
    @ti.func
    def quat_to_matrix(self, q: ti.template()) -> ti.Matrix:
        rotation = ti.Matrix([[0]* 3 for _ in  range(3)], ti.f32)
        w = q[None][0]
        x = q[None][1]
        y = q[None][2]
        z = q[None][3]
        rotation[0,0] = 1 - 2*y*y  - 2*z*z
        rotation[0,1] = 2*x*y - 2*z*w
        rotation[0,2] = 2*x*z + 2*y*w
        rotation[1,0] = 2*x*y + 2*z*w
        rotation[1,1] = 1 - 2*x*x - 2*z*z
        rotation[1,2] = 2*y*z - 2*x*w
        rotation[2,0] = 2*x*z - 2*y*w
        rotation[2,1] = 2*y*z + 2*x*w
        rotation[2,2] = 1 - 2*x*x - 2*y*y
        # rotation[3,3] = 1
        return rotation
        
    @ti.func
    def magnitude(vector): 
        return ti.sqrt(sum(pow(element, 2) for element in vector))

    @ti.func
    def Get_Cross_Matrix(self, a) -> ti.Matrix:
        A = ti.Matrix([[0] * 3 for _ in range(3)], ti.f32)      
        A[0, 0] = 0
        A[0, 1] = -a [2]
        A[0, 2] = a [1]
        A[1, 0] = a [2]
        A[1, 1] = 0
        A[1, 2] = -a [0]
        A[2, 0] = -a [1]
        A[2, 1] = a [0]
        A[2, 2] = 0
        # A[3, 3] = 1
        return A

    @ti.kernel
    def Collision_Impulse(self, P: ti.template(), N:ti.template()):
        R = self.quat_to_matrix(self.q)
        sumn = 0
        sumx = ti.Vector([0.0, 0.0, 0.0])
        for i in range(0, self.vertices.shape[0]):
            Rri = R @ self.vertices[i]
            x_i = self.position[None] + Rri
            # print( ti.Vector.dot(x_i - P, N))
            if ti.Vector.dot(x_i - P, N) < 0:
                print("Error")
                v_i = self.velocity[None]+ti.Vector.dot(self.w[None], Rri);
                if ti.Vector.dot(v_i, N) < 0:
                    ti.atomic_add(sumx, self.vertices[i])
                    ti.atomic_add(sumn, 1)
                    # print(sumn)
        # pass
        if sumn != 0:
            r_collision = sumx/sumn
            # print("r_col", r_collision)
            Rr_collision = R @ r_collision
            v_collision = self.velocity[None] + ti.Vector.cross(self.w[None], Rr_collision)
            I = R @ self.I_ref @ R.transpose()
            v_N = ti.Vector.dot(v_collision, N)*N
            v_T = v_collision - v_N
            
            a = max(1.0 - 0.5 * (1.0 + 0.3) * v_N.norm() / v_T.norm(), 0.0);
            v_N_new = -1.0 * 0.3 * v_N
            v_T_new = a * v_T
            v_new = v_N_new + v_T_new
            R_r_cross = self.Get_Cross_Matrix(Rr_collision)
            identity_matrix = ti.Matrix.identity(dt=ti.f32, n=3)
            K = identity_matrix * (1.0/self.vertices.shape[0]) - R_r_cross*I.inverse()*R_r_cross
            j = K.inverse() @ (v_new - v_collision)
            print(self.velocity[None])
            self.velocity[None] = self.velocity[None] + 1.0/self.vertices.shape[0]*j
            print(self.velocity[None])
            self.w[None] += I.inverse() @ (R_r_cross @ j)
        
    def update(self):
        for i in range(3):
            self.velocity[None][i] = (self.velocity[None][i] + self.dt/2 * self.g[None][i])*self.linear_decay
            self.w[None][i]        *= self.linear_decay
        self.Collision_Impulse(ti.Vector([0, 0.01, 0]), ti.Vector([0, 1, 0]))
        # self.Collision_Impulse(ti.Vector([2, 0, 0]), ti.Vector([-1, 0, 0]))
        if self.velocity[None].norm() < 0.01:
            self.times -= 1
            if self.times == 0:
                self.velocity *= 0.01
                self.w *= 0.01
                return
        else:
            self.times = 10
        # dw = 0.5 * self.dt * self.w[None]
        qw = [0.5*self.dt*self.w[None][0], 0.5*self.dt*self.w[None][1],0.5*self.dt*self.w[None][2], 0.0]
        # print(self.q[None])
        self.q[None] = self.q[None] + qw*self.q[None]
        for i in range(3):
            self.position[None][i] = self.position[None][i] + self.velocity[None][i] * self.dt
        self.dump()
        
    @ti.kernel
    def dump(self):
        R = self.quat_to_matrix(self.q)
        for i in range(0, self.vertices.shape[0]):
            Rri = R @ self.vertices[i]
            x_i = self.position[None] + Rri
            self.x[i] = x_i
q = Bunny("/home/star/workspace/personal/Ti-SPH/data/models/bunny.obj")

window = ti.ui.Window('SPH', (1024, 1024), show_window = True, vsync=False)
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(5.5, 2.5, 4.0)
camera.up(0.0, 1.0, 0.0)
camera.lookat(-1.0, 0.0, 0.0)
camera.fov(70)
scene.set_camera(camera)
# x1 = ti.Vector([0, -1, 0])
# print(ti.Vector.dot(x1, ti.Vector([])))

# q.update()
    
if __name__ == "__main__":
    while window.running:
        for i in range(5):
            q.update()
        particle_info = q.vertices
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
        # positions = ti.Vector.field(3, dtype=ti.f32, shape= len(particle_info['position']))
        # for i in range(len(particle_info['position'])):
        #     for j in range(3):
        #         positions[i][j] = particle_info['position'][i][j]
        scene.particles(q.x, color = (0.68, 0.26, 0.19), radius = 0.01)
        # print(particle_info['position'])

        canvas.scene(scene)
        window.show()
