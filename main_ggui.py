import taichi as ti
import json
from core.partice_system import partice_systemv3, partice_systemv4
from core.sph.wcsph import WCSPH
ti.init(arch=ti.gpu,debug=True, cpu_max_num_threads=1, advanced_optimization=False)

N = 3

particles_pos = ti.Vector.field(3, dtype=ti.f32, shape = N)
points_pos = ti.Vector.field(3, dtype=ti.f32, shape = N)

@ti.kernel
def init_points_pos(points : ti.template()):
    for i in range(points.shape[0]):
        points[i] = [i for j in ti.static(range(3))]

init_points_pos(particles_pos)

window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(5, 2, 2)
box_anchors = ti.Vector.field(3, dtype=ti.f32, shape = 8)
box_anchors[0] = ti.Vector([0.0, 0.0, 0.0])
box_anchors[1] = ti.Vector([0.0, 3.0, 0.0])
box_anchors[2] = ti.Vector([5.0, 0.0, 0.0])
box_anchors[3] = ti.Vector([5.0, 3.0, 0.0])

box_anchors[4] = ti.Vector([0.0, 0.0, 2.0])
box_anchors[5] = ti.Vector([0.0, 3.0, 2.0])
box_anchors[6] = ti.Vector([5.0, 0.0, 2.0])
box_anchors[7] = ti.Vector([5.0, 3.0, 2.0])

box_lines_indices = ti.field(int, shape=(2 * 12))
for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
    box_lines_indices[i] = val
    
if __name__ == "__main__":
    with open("./data/scenes/demo_3d.json", "r") as f:
        simulation_config = json.load(f)
    
    config = simulation_config['configuration']
    ps = partice_systemv3.ParticleSystemV3((100, 100, 100), simulation_config)
    # add fluid and rigid
    ps.add_fluid_and_rigid()
    wcsph = WCSPH(ps)
    while window.running:
        for i in range(5):
            wcsph.step()
        particle_info = ps.dump()
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        positions = ti.Vector.field(3, dtype=ti.f32, shape= len(particle_info['position']))
        for i in range(len(particle_info['position'])):
            for j in range(3):
                positions[i][j] = particle_info['position'][i][j]
        # scene.particles(positions, color = (0.68, 0.26, 0.19), radius = 0.01)
        canvas.scene(scene)
        window.show()
        