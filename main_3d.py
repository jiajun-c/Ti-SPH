import taichi as ti
import json
from core.partice_system import partice_systemv4
from core.sph.wcsphv2 import WCSPHV2
from utils.lines import getlines
ti.init(arch=ti.cuda)

window = ti.ui.Window('SPH', (1024, 1024), show_window = True, vsync=False)
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(5.5, 2.5, 4.0)
camera.up(0.0, 1.0, 0.0)
camera.lookat(-1.0, 0.0, 0.0)
camera.fov(70)
scene.set_camera(camera)

if __name__ == "__main__":
    with open("./data/scenes/demo_3d.json", "r") as f:
        simulation_config = json.load(f)
    points, indices = getlines(simulation_config['configuration'])
    print(points)
    config = simulation_config['configuration']
    ps = partice_systemv4.ParticleSystemV4(simulation_config)
    # add fluid and rigid
    # ps.add_fluid_and_rigid()
    wcsph = WCSPHV2(ps)
    # wcsph.step()
    # wcsph.step()
    while window.running:
        for i in range(5):
            wcsph.step()
        particle_info = ps.dump()
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
        # positions = ti.Vector.field(3, dtype=ti.f32, shape= len(particle_info['position']))
        # for i in range(len(particle_info['position'])):
        #     for j in range(3):
        #         positions[i][j] = particle_info['position'][i][j]
        scene.particles(ps.x, color = (0.68, 0.26, 0.19), radius = 0.01)
        # print(particle_info['position'])
        scene.lines(points, width=1.0, indices=indices,  color = (0.99, 0.68, 0.28))

        canvas.scene(scene)
        window.show()
    