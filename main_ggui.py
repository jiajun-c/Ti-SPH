import taichi as ti
import json
from core.partice_system import partice_systemv3
from core.sph.wcsph import WCSPH
ti.init(arch=ti.cpu)


window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(5, 2, 2)

if __name__ == "__main__":
    with open("./data/scenes/demo_3d.json", "r") as f:
        simulation_config = json.load(f)
    
    config = simulation_config['configuration']
    ps = partice_systemv3.ParticleSystemV3((16, 16, 16), simulation_config)
    # add fluid and rigid
    ps.add_fluid_and_rigid()
    wcsph = WCSPH(ps)
    while window.running:
        for i in range(5):
            print("Running")
            wcsph.step()
        particle_info = ps.dump()
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        scene.particles(particle_info, color = (0.68, 0.26, 0.19), radius = 0.2)
        # Draw 3d-lines in the scene
        # scene.lines(points_pos, color = (0.28, 0.68, 0.99), width = 5.0)
        canvas.scene(scene)
        window.show()
        