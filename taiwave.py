import taichi as ti
import constants as cst

# single monochromatic wave
@ti.dataclass
class Wave:
    w: ti.f64
    phi0: ti.f64 # initial phase of whistler wave
    psi: ti.f64 # wave vector angle 
    Ewx: ti.f64
    Ewy: ti.f64
    Ewz: ti.f64
    Bwx: ti.f64
    Bwy: ti.f64
    Bwz: ti.f64

    k: ti.f64  # k value, not vector
    k_vector: ti.types.vector(3, ti.f64)
    Bw: ti.types.vector(3, ti.f64)
    Ew: ti.types.vector(3, ti.f64)

    phi: ti.f64  # wave phase

    @ti.func
    def initialize(self, w, phi0, Ewx, Ewy, Ewz, Bwx, Bwy, Bwz, k, psi):
        self.w = w
        self.phi0 = phi0
        self.Ewx = Ewx
        self.Ewy = Ewy
        self.Ewz = Ewz
        self.Bwx = Bwx
        self.Bwy = Bwy
        self.Bwz = Bwz
        self.k = k
        self.psi = psi
        self.k_vector = [self.k * ti.sin(self.psi), 0, self.k * ti.cos(self.psi)]

    @ti.func
    def get_wavefield(self, t, r):
        """Get the wave field at the location r and time t

        Args:
            t (_type_): _description_
            r (_type_): _description_

        Returns:
            _type_: _description_
        """
        # calculate the wave phase
        # because uniform background, do not need the integral
        phi = self.k_vector.dot(r) - self.w * t + self.phi0
        # print('t',t,'phi',phi,'sinphi',ti.sin(phi))
        # print('k',self.k,'r',r)
        # print('t',t,'phi',phi,'sinphi',ti.sin(phi))
        self.Bw = [self.Bwx * ti.cos(phi), -1 * self.Bwy * ti.sin(phi), self.Bwz * ti.cos(phi)]

        self.Ew = [-1 * self.Ewx * ti.sin(phi), -1 * self.Ewy * ti.cos(phi), -1 * self.Ewz * ti.sin(phi)]

        self.phi = phi
