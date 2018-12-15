from fluid import *
from viz import Renderer


# r = Renderer(get_vortex_sim(), interval=None)
r = Renderer(ShearScenario.new(), interval=None)

r.start_render()
