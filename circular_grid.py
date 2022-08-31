import acoular
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