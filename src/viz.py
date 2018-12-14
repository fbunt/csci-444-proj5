import matplotlib as mpl
# Work around for recent bug that causes crash. Must happend before
# plt imported
mpl.use("TkAgg")
from matplotlib import animation
import matplotlib.pyplot as plt
import time

from fluid import get_test_simulation


plt.style.use("dark_background")


def get_data(sim):
    # TODO: finalize
    return sim.rho


class Renderer:
    def __init__(self, sim, fps=60, zmin=0, zmax=2):
        self.sim = sim
        self.zmin = zmin
        self.zmax = zmax
        self.fps = fps
        self.interval = 1.0 / fps

        self.fig = None
        self.img = None
        self.ani = None
        self.elapsed = 0
        self.last = time.time()

    def start_render(self):
        plt.ion()
        self.fig = plt.figure()
        self.img = plt.imshow(
            get_data(self.sim), vmin=self.zmin, vmax=self.zmax, animated=True
        )
        self.ani = animation.FuncAnimation(
            self.fig,
            self._render,
            init_func=self._init_render,
            interval=10,
        )

    def _init_render(self):
        self.sim.add_particle(0, 0, 0, 0, 0.2)
        self.img.set_array(get_data(self.sim))
        return self.img

    def _render(self, frame):
        last = time.time()
        elapsed = last - self.last
        print(f"Elapsed: {elapsed}")
        self.last = last
        self.sim.time_step()
        self.img.set_data(get_data(self.sim))
