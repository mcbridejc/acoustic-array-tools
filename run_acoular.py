#%%
# 


%matplotlib widget
import acoular
import h5py
import matplotlib.pyplot as plt
import numpy as np
from micarray import array_x5, array_x6


mic_grid = acoular.microphones.MicGeom()
mic_grid.mpos_tot = np.array(array_x5).T / 1000

from os import path

from traits.api import HasPrivateTraits, Float, Property, Any, \
property_depends_on, cached_property
import math

class CircularGrid(acoular.grids.Grid):
    radius = Float(1.0, desc="Radius of circle")
    increment = Float(1.0, desc="Angular increment in deg")
    z = Float(0.0, desc="Z position")
    # internal identifier
    digest = Property(
        depends_on = ['radius', 'increment', 'z']
        )
    #: Number of grid points along circle, readonly.
    nsteps = Property(
        desc="number of grid points along circle")
    
    @property_depends_on('increment')
    def _get_nsteps(self):
        return int(round(360/self.increment))
    
    @property_depends_on('nsteps')
    def _get_shape ( self ):
        return (self.nsteps,)

    @property_depends_on('nsteps')
    def _get_size ( self ):
        return self.nsteps
    
    @property_depends_on('radius, increment')
    def _get_gpos(self):
        n_steps = self.nsteps
        points = np.zeros((3, n_steps))
        for i in range(n_steps):
            alpha = 2 * math.pi * i / n_steps
            x = math.sin(alpha) * self.radius
            y = math.cos(alpha) * self.radius
            points[:, i] = (x, y, self.z)
        return points

    @cached_property
    def _get_digest( self ):
        return acoular.internal.digest( self )

    def extend(self):
        return (-self.radius, self.radius, -self.radius, self.radius)
    
# micgeofile = path.join(path.split(acoular.__file__)[0],'xml','array_64.xml')
# mic_grid = acoular.MicGeom( from_file=micgeofile )

# make a simulated source
# n1 = acoular.SineGenerator(sample_freq=48000, freq=500, numsamples=100000)
# p1 = acoular.PointSource(signal=n1, mics=mic_grid,  loc=(0.5, 0.5, 0.01))
# wh5 = acoular.WriteH5(source=p1, name='sim_series.h5')
# wh5.save()
# del wh5
# ts = acoular.TimeSamples(name='sim_series.h5')

ts = acoular.TimeSamples(name='tone_90deg.h5')

ps = acoular.PowerSpectra(time_data=ts, block_size=1024, window='Hanning')
#grid = acoular.RectGrid(x_min=-0.8, x_max=0.8, y_min=-0.8, y_max=0.8, z=0.01, increment=0.05)
grid = CircularGrid(z=0.01, increment=5)
st = acoular.SteeringVector(grid=grid, mics=mic_grid)
bb = acoular.BeamformerBase(freq_data=ps, steer=st)
pm = bb.synthetic(500, 1)
print("Solving...")
Lm = acoular.L_p( pm )

Lm
# %%
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
plt.plot(np.arange(0, 360, grid.increment) * math.pi / 180., Lm - Lm.min())
ax.set_theta_zero_location('N')
# plt.imshow(Lm.T, origin='lower', vmax = 101, vmin=97, extent=grid.extend(), interpolation='bicubic')
# plt.colorbar()
# %%
