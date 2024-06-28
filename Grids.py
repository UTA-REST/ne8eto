import numpy as np


def MaxBoltz2D(mkT,pr,pz,pr0=0,pz0=0):
    return 2.*np.pi*(2*np.pi*mkT)**(-3/2)*np.exp(-((pr-pr0)**2+(pz-pz0)**2)/(2.*mkT))

def NullDist2D(coord):
    return np.zeros_like(coord)

# The data grid on which the evolution will take place.
#  Pre-calc a bunch of geometric things to help go fast.
class Grid2D:

    # limits of grid
    zlim=0
    rlim=0
    rlim2=0

    # flat range of R and Z bins
    RRange=[]
    ZRange=[]

    XX,YY,ZZ=[], [], []

    # 2D meshes
    RR2=[]
    ZZ2=[]
    RSph2=[]
    th2=[]

    # flat 2d meshes
    RR2f=[]
    ZZ2f=[]

    #zlim defines how many grid points.
    def __init__(self,zlim):
        self.zlim=zlim
        self.rlim=int(zlim*np.sqrt(2)+1)
        self.rlim2=self.rlim**2
        self.RRange=range(0,self.rlim)
        self.ZRange=range(-zlim,zlim)

        self.xyzlim=[zlim,zlim,zlim]

        self.RR2,self.ZZ2=np.meshgrid(self.RRange,self.ZRange)
        self.RR2f=self.RR2.ravel()
        self.ZZ2f=self.ZZ2.ravel()

        self.XX,self.YY,self.ZZ=np.meshgrid(self.ZRange,self.ZRange,self.ZRange)
        self.rlim2=self.rlim**2

        self.RSph2=(self.RR2**2+self.ZZ2**2)**0.5
        self.th2=np.arctan2(self.ZZ2,self.RR2)


# The distribution base class.
#  includes momentum and velocity scale factors and a null dist

class Distribution:
    f=None
    grid=None
    pscale=None
    vscalefactor=None
    pscalefactor=None

    # Get velocity conv factor in ms^-1
    def VScale_SI(self):
        return self.pscale* self.vscalefactor

    def PScale_SI(self):
        return self.pscale*self.pscalefactor

    def PScale_eV(self):
        return self.pscale

    def __init__(self, pscale=1, gridpoints=30, m=3e9):
        self.grid=Grid2D(gridpoints)
        self.pscale=pscale
        self.vscalefactor=3e8/m
        self.pscalefactor=1.6e-19
        self.f=NullDist2D(self.grid.RR2)





# An example distribution - this one is a sliced maxwell boltzmann.

class SlicedMB2D(Distribution):
    def __init__(self, T=10, mkt=80, gridpoints=30, m=3e9, RCut=100,ZCut=100,Rho=6e19,pz0=0,pr0=0):
        kT=T*8.62e-5 # in eV
        Distribution.__init__(self,pscale=np.sqrt((m*kT)/mkt), gridpoints=gridpoints, m=m)
        mb2d=Rho*MaxBoltz2D(mkt,self.grid.RR2,self.grid.ZZ2,pz0=pz0,pr0=pr0)
        CutMask=(self.grid.RR2<RCut)*(self.grid.ZZ2<ZCut)
        self.f=mb2d*CutMask
