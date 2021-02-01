import sys
sys.path.append('build')


import MatterSim
import math

sim = MatterSim.Simulator()
sim.setRenderingEnabled(False)
sim.setDiscretizedViewingAngles(True)
sim.setCameraResolution(160, 120)
# sim.setPreloadingEnabled(True)
sim.setDepthEnabled(True)
sim.setCameraVFOV(math.radians(60))
sim.initialize()
sim.newEpisode(['2t7WUuJeko7'], ['1e6b606b44df4a6086c0f97e826d4d15'], [0], [0])
state = sim.getState()
import pdb; pdb.set_trace()
