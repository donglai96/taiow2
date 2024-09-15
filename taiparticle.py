import taichi as ti
import constants as cst

@ti.dataclass
class Particle:
    r:ti.types.vector(3,ti.f64) # location
    p:ti.types.vector(3,ti.f64) # momentum
    m: ti.f64 #mass
    q: ti.f64 # charge
    t: ti.f64 # simulation time for the particle
    alpha: ti.f64 # record pitch angle
    alpha0: ti.f64 # initial pitch angle
    phi: ti.f64 # record particle gyro phase

    Ep: ti.types.vector(3,ti.f64) # electric field at particle location
    Bp: ti.types.vector(3,ti.f64) # magnetic field at particle location

    @ti.func
    def initParticles(self, mm, qq):
        """Init the particles with mass and charge,
        it is important to init objects in ti.kernel

        Args:
            mm (_type_): mass
            qq (_type_): charge
        """
        self.m = mm
        self.q = qq
        self.t = 0
        self.phi = 0 # initial gyrophase
        self.Ep = ti.Vector([0.0,0.0,0.0])
        self.Bp = ti.Vector([0.0,0.0,0.0])

        
    @ti.func
    def initPos(self, x, y, z):
        """Initialize the position of the particle.
        
        Args:
            x (ti.f64): x-coordinate of the position.
            y (ti.f64): y-coordinate of the position.
            z (ti.f64): z-coordinate of the position.
        """
        self.r = ti.Vector([x, y, z])
    
    @ti.func
    def initMomentum(self, px, py, pz):
        """Initialize the momentum of the particle.
        
        Args:
            px (ti.f64): x-component of the momentum.
            py (ti.f64): y-component of the momentum.
            pz (ti.f64): z-component of the momentum.
        """
        self.p = ti.Vector([px, py, pz])
        
    @ti.func
    def get_pitchangle(self):
        """Calculate the pitch angle of the particle."""
        self.alpha = ti.acos(self.p[2] / self.p.norm())
        
    @ti.func
    def boris_push(self, dt, E, B):
        """Push the particles using Boris' method.
        
        Update the velocity of particles.
        
        Args:
            dt (ti.f64): Time step.
            E (ti.types.vector(3, ti.f64)): Electric field at particle location.
            B (ti.types.vector(3, ti.f64)): Magnetic field at particle location.
        """
        p_minus = self.p + self.q * E * dt / 2.0
 
        gamma = ti.sqrt(1 + p_minus.norm()**2 / (self.m**2 * cst.C**2))
        
        t = self.q * dt * B / (2 * gamma * cst.C * self.m)  # Gaussian unit
        
        p_p = p_minus + p_minus.cross(t)

        s = 2 * t / (1 + t.dot(t))

        p_plus = p_minus + p_p.cross(s)

        self.p = p_plus + self.q * E * dt / 2.0
        
    @ti.func
    def leap_frog(self, dt, E, B):
        """Perform leap frog integration to update the particle's position and velocity.
        
        Args:
            dt (ti.f64): Time step.
            E (ti.types.vector(3, ti.f64)): Electric field at particle location.
            B (ti.types.vector(3, ti.f64)): Magnetic field at particle location.
        """
        self.boris_push(dt, E, B)
        gammam = self.m * ti.sqrt(1 + self.p.norm()**2 / (self.m**2 * cst.C**2))
        
        v_xyz = self.p / gammam
        
        self.r += dt * v_xyz
