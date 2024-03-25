import taichi as ti
import argparse
from mpi4py import MPI
import taichi_init
import json

def main(show_gui):
    comm = MPI.COMM_WORLD
    count = comm.size   
    rank = comm.rank
    # 初始化设备显卡信息
    taichi_init.device_init()
    if rank == 0 and show_gui == True:
        with open("./data/scenes/demo_fluid_and_rigid.json", "r") as f:
            simulation_config = json.load(f)
        window = ti.ui.Window('SPH', (1024, 1024), show_window = True, vsync=False)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(5.5, 2.5, 4.0)
        camera.up(0.0, 1.0, 0.0)
        camera.lookat(-1.0, 0.0, 0.0)
        camera.fov(70)
        scene.set_camera(camera)
    # 在根节点上做一些渲染的工作？
    while True:
        if rank == 0 and show_gui == True:
            canvas.scene(scene)
            window.show()
            

if __name__ == '__main__':
    main()