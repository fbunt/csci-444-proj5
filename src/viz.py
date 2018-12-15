from collections import namedtuple
import pendulum as pdm
import matplotlib as mpl

# Work around for recent bug that causes crash. Must happend before
# plt imported
mpl.use("TkAgg")
from matplotlib import cm
from matplotlib import animation
import matplotlib.pyplot as plt
from queue import Queue, Empty
import threading
import time


plt.style.use("dark_background")


_Message = namedtuple("Message", ("kill", "pause"))


# Use these to make errors harder. Less to remember
def _get_kill_msg():
    return _Message(True, False)


def _get_pause_msg(pause=True):
    return _Message(False, pause)


_Data = namedtuple("Data", ("data", "time", "iteration"))


def _sim_worker(msg_q, result_q, stepper):
    """
    Continually calculates the next step and pushes the results to `result_q`
    """
    stop = False
    pause = False
    while not stop:
        # Check for messages
        try:
            m = msg_q.get_nowait()
            stop = m.kill
            pause = m.pause
            msg_q.task_done()
        except Empty:
            pass

        # Pause until we hear otherwise
        # Use blocking to sleep the thread
        while pause:
            m = msg_q.get(block=True)
            stop = m.kill
            pause = m.pause
            if stop:
                break

        # Do work
        if not stop:
            try:
                stepper.step()
                data = stepper.data().copy()
                time = stepper.time
                iteration = stepper.iteration
                d = _Data(data, time, iteration)
                result_q.put_nowait(d)
            except Exception as e:
                print(e, flush=True)


class Renderer:
    def __init__(self, sim, interval=50, zmin=0, zmax=1.0, cmap=cm.magma):
        self.sim = sim
        self._zmin = zmin
        self._zmax = zmax
        if interval is None:
            # Go as fast as we can
            interval = 1
        self.interval = interval

        self._cmap = cmap
        self._fig = None
        self._it_text = None
        self._time_text = None
        self._ax = None
        self._img = None
        self._ani = None

        # Flag to tell if paused or not
        self._running = False
        self._last = time.time()
        # Used to kill or pause simulation worker thread
        self._msg_q = Queue()
        # Used to get data back from simulation thread
        self._data_q = Queue()
        # Background thread for simulation work.
        self._work_thread = None
        self.last_data = None
        # Initial data
        self.last_data = _Data(self.sim.data(), 0, 0)

    def start_render(self):
        """
        Begins running and rendering the simulation. Can be used to resume
        again after stopping.
        """
        self._init_render()
        try:
            self._ani = animation.FuncAnimation(
                self._fig,
                self._render,
                init_func=self._anim_init,
                interval=self.interval,
            )
            print("Starting work thread")
            self._work_thread.start()
            plt.savefig("first_frame.png", dpi=150)
            plt.show()
        finally:
            # Clean up
            print("Killing work thread")
            while not self._msg_q.empty():
                self._msg_q.get_nowait()
            # Make sure that the message isn't left in the queue, to the best
            # of our ability
            try:
                self._msg_q.put(_get_kill_msg(), block=True, timeout=2)
            except Empty:
                print("Could not cleanly kill work thread")
            self._work_thread = None

    def _init_render(self):
        # Create background thread for simulation processing
        self._work_thread = threading.Thread(
            target=_sim_worker, args=(self._msg_q, self._data_q, self.sim)
        )
        self._fig, self._ax = plt.subplots()
        self._img = plt.imshow(
            self.sim.data(),
            vmin=self._zmin,
            vmax=self._zmax,
            animated=True,
            cmap=self._cmap,
        )
        self._it_text = self._ax.text(
            0,
            1.05,
            _get_iteration_str(self.last_data.iteration),
            horizontalalignment="left",
            verticalalignment="top",
            transform=self._ax.transAxes,
            fontdict={"color": "w", "fontsize": 15},
        )
        self._time_text = self._ax.text(
            1,
            1.05,
            _get_time_str(self.last_data.time),
            horizontalalignment="right",
            verticalalignment="top",
            transform=self._ax.transAxes,
            fontdict={"color": "w", "fontsize": 15},
        )
        cbar = self._fig.colorbar(self._img)
        cbar.ax.set_ylabel("Density")
        # Pause/unpause on canvas click
        self._fig.canvas.mpl_connect("button_press_event", self._pause_toggle)
        self._running = True

    def _anim_init(self):
        self._img.set_array(self.sim.data())
        return self._img

    def _render(self, frame):
        last = time.time()
        elapsed = last - self._last
        print("FPS: {:5.2f}".format(1 / elapsed))
        self._last = last
        # Grab next data slice to render
        data = self._data_q.get(block=True)
        self._img.set_data(data.data)
        plt.setp(self._it_text, text=_get_iteration_str(data.iteration))
        plt.setp(self._time_text, text=_get_time_str(data.time))
        return self._img

    def _pause_toggle(self, event):
        if self._running:
            self._ani.event_source.stop()
            self._msg_q.put_nowait(_get_pause_msg(True))
            self._running = False
        else:
            self._ani.event_source.start()
            self._msg_q.put_nowait(_get_pause_msg(False))
            self._running = True


def _get_time_str(seconds):
    dur = pdm.duration(seconds=seconds)
    ms = int(dur.microseconds * 1e-3)
    tstr = (
        f"{dur.hours:02}:{dur.minutes:02}:{dur.remaining_seconds:02}:{ms:03}"
    )
    return tstr


def _get_iteration_str(it):
    return f"It:{it:10d}"
