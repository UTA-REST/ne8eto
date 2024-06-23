import numpy as np
import copy
from random import choices, getrandbits
from Grids import *


#Comments for future self:
#. dist.f should be equal to P(pr,phi) but where there is no phi dependence
#. Thus for total number of particles at given r, P(r) = 2*pi*r*dist(f)
#. 
#. When dfdt is calculated, it applies to each P(pr,phi) equivalently, since p4 is an input

class Evolver2D:

    grid=None
    verbose=None
    CrossSec=None

    def __init__(self,grid,CrossSec=6.15e-20,verbose=False):
        self.grid=grid
        self.verbose=verbose
        self.CrossSec=CrossSec


    def GetDfDtFast(self, dist, col):
        #Prepare output array
        dfdt2=np.zeros_like(dist.f)
        counter=np.zeros_like(dist.f)

        pbins=len(dist.f.ravel())
        arraylen=col*pbins

        #each of these signs is applied to a randomly generated angle,
        randsigns2=(-1)**(np.random.randint(0,2,size=arraylen))
        randsigns3=(-1)**(np.random.randint(0,2,size=arraylen))

        # Since these signs apply to random angles we don't necessarily
        #  need them totally random. This is faster.
        #randsigns2=(-1)+2*(np.arange(0,col*pbins,1)>int(arraylen/2))
        #randsigns3=(-1)+2*(np.arange(0,col*pbins,1)%2==1)

        # To understand this norm factor see Luiten, Phys Rev A 53, 1 (1995) Eq. 10
        normfactor=(2.*self.grid.zlim)**3   *   dist.VScale_SI()  *  self.CrossSec/(2*np.pi)  *      (4.*np.pi)                *  (2.*np.pi)
        #           [volume for p3 int]         [scaling for q/m]     [xs/2pi in prefactor]      [volume for dOmega int]    [converting f2D to f3D]

        # we can pick coord system such that p4 defines x axis, WLOG.
        # We can also optimize p4 sampling however we like,  because we divide out the samples per bin in the end.
        #  Here we emphasize lower r values by sampling as r^0.5, since we expect our phase space disnt is more concentrated at low r.
        p4=np.array([np.random.uniform(0,self.grid.rlim**0.5,size=arraylen)**2, np.zeros(arraylen),np.random.uniform(-self.grid.zlim,self.grid.zlim,size=arraylen)]).T
        Z4=np.array(p4[:,2]+self.grid.zlim).astype('int')
        R4=np.array(p4[:,0]).astype('int')

        # Generate random p3s
        p3=np.array([np.random.uniform(-self.grid.zlim,self.grid.zlim,size=arraylen) for i in (0,1,2)]).T

        P=(p3+p4)/2                               # COM momentum/2


        cph_out=2*np.random.random(arraylen)-1         # random angles for 12 in c.o.m.
        sph_out=(1-cph_out**2)**0.5*randsigns2
        cth_out=2*np.random.random(arraylen)-1
        sth_out=(1-cth_out**2)**0.5*randsigns3

        qmag=sum((p4-p3).T**2)**0.5/2                                      # mom xfer/2  mag
        dp=(qmag*np.array([sth_out*cph_out,sth_out*sph_out,cth_out])).T    # mom xfer/2  3-vec

        p2=P+dp   # initial moms
        p1=P-dp

        for i in range(0,len(p3)):
            # Calculate Boltzmann source and sink terms
            term1, term2=self.GetBoltzmannTerms2D(p1[i],p2[i],p3[i],p4[i],dist.f)
            counter[Z4[i],R4[i]]+=1
            if(np.abs(term1-term2)>1e-20):
                dfdt2[Z4[i],R4[i]]+=(qmag[i])*(term1-term2)

        eps=1e-10
        return(dfdt2  / (counter+eps) * normfactor)


    def GetGradP(self, f2D):
        GradZ=f2D[2:,:]-f2D[:-2,:]
        GradRho=f2D[:,1:]-f2D[:,:-1]

    # Calculate source and sink terms in Boltzmann based on dist function f
    def GetBoltzmannTerms2D(self,p1,p2,p3,p4, f2D):
        term1=0
        term2=0


        # check all p's are within grid, otherwise source and sink both 0.
        #  go in steps for efficiency. p1, p2 are most likely fails, so do these first.
        R1_2=p1[0]**2+p1[1]**2
        if((R1_2<self.grid.rlim2) and (p1[2]<self.grid.zlim) and (p1[2]>=-self.grid.zlim)):
            R2_2=p2[0]**2+p2[1]**2
            if((R2_2<self.grid.rlim2) and (p2[2]<self.grid.zlim) and (p2[2]>=-self.grid.zlim)):
                term1=f2D[int(p1[2]+self.grid.zlim),int(R1_2**0.5)]*f2D[int(p2[2]+self.grid.zlim),int(R2_2**0.5)]
        R3_2=p3[0]**2+p3[1]**2
        if((R3_2<self.grid.rlim2) and (p3[2]<self.grid.zlim) and (p3[2]>=-self.grid.zlim)):
            R4_2=p4[0]**2+p4[1]**2
            if((R4_2<self.grid.rlim2) and (p4[2]<self.grid.zlim) and (p4[2]>=-self.grid.zlim)):
                term2=f2D[int(p3[2]+self.grid.zlim),int(R3_2**0.5)]*f2D[int(p4[2]+self.grid.zlim),int(R4_2**0.5)]

        return term1, term2


    #Thermalize an initial state distribution
    def ThermalizeIt(self,dist, times=np.arange(0,0.002,0.0002),verbose=False):
        fs=[]
        dfdts=[]

        for ti in range(0,len(times)-1):
            if(verbose):
                print("Starting time step "+ str(ti)+ " of " +  str(len(times)-1) )
            fs.append(copy.deepcopy(dist.f))
            dfdt=self.GetDfDtFast(dist, 500)
            StepTime=times[ti+1]-times[ti]
            dfdts.append(copy.deepcopy(dfdt))
            dist.f=dist.f+dfdt*StepTime
        return fs, dfdts

    # Apply "wiggling" in phase space
    def CrescentIt(self,crescentangle, dist,smears=1000):
        newf=np.zeros_like(dist.f)
        flatf=dist.f.ravel()
        for i in range(0,len(flatf)):
            if(flatf[i]>(1e-8*np.max(flatf))):
                Angle=np.random.uniform(-1,1,smears)*crescentangle

                th=dist.grid.th2.ravel()[i]
                R=dist.grid.RSph2.ravel()[i]
                for a in Angle:
                    newth=th-a
                    newRR=int(np.abs(R*np.cos(newth)))
                    newZZ=int(np.round(R*np.sin(newth),0))
                    if((newRR<dist.grid.rlim-1) and (newRR>=0) and (newZZ<(dist.grid.zlim-1)) and (newZZ>-dist.grid.zlim)):
                        newf[newZZ+dist.grid.zlim,newRR]+=flatf[i]/smears
        dist.f=newf
        return dist


    #Cool an initial state distribution according to some cut function
    def CoolIt(self,dist, CutFunction, timesteps=10,StepFactor=0.1,cols=500):
        fs=[]
        dfdts=[]
        times=[0]
        energyfracovercut=[]
        particlefracovercut=[]
        c2=self.grid.RR2**2+self.grid.ZZ2**2

        for ti in range(0,timesteps):
            CutMask=CutFunction(times[-1],self.grid)
            InvCutMask=np.ones_like(CutMask)-CutMask.astype('int')
            dist.f=dist.f*CutMask
            fs.append(copy.deepcopy(dist.f))

            dfdt=self.GetDfDtFast(dist, cols)
            StepTime=np.max(np.abs(dist.f))/np.max(np.abs(dfdt))*StepFactor

            particlefracovercut.append(sum(sum(dfdt*InvCutMask*self.grid.RR2))/sum(sum(dist.f*self.grid.RR2))*StepTime)
            energyfracovercut.append(sum(sum(c2*dfdt*InvCutMask*self.grid.RR2))/sum(sum(c2*dist.f*self.grid.RR2))*StepTime)
            dfdts.append(copy.deepcopy(dfdt))
            dist.f=(dist.f+dfdt*StepTime)*CutMask

            times.append(times[-1]+StepTime)
            if(self.verbose):
                print("Cooling step "+ str(ti)+", t="+str(times[-1])+", dp="+str(particlefracovercut[-1])+", dE="+str(energyfracovercut[-1]))
        ETraj=np.cumprod(1-np.array(energyfracovercut))
        PTraj=np.cumprod(1-np.array(particlefracovercut))

        return fs, dfdts,times,ETraj,PTraj



    #Cool an initial state distribution according to some cut function while wiggling
    def WiggleCoolIt(self, dist, CutFunction, timesteps=10,StepFactor=0.1,cols=500,crescentangle=1.1):
        fs=[]
        dfdts=[]
        times=[0]
        energyfracovercut=[]
        particlefracovercut=[]
        cutpos=[]
        c2=self.grid.RR2**2+self.grid.ZZ2**2

        for ti in range(0,timesteps):

            cutpos.append(20-int(ti/2))
            CutMask=CutFunction(times[-1],grid)
            InvCutMask=np.ones_like(CutMask)-CutMask.astype('int')
            dist=self.CrescentIt(crescentangle,dist)
            dist.f=dist.f*CutMask
            fs.append(copy.deepcopy(dist.f))

            dfdt=self.GetDfDtFast(dist, cols)
            StepTime=np.max(np.abs(dist.f))/np.max(np.abs(dfdt))*StepFactor

            particlefracovercut.append(sum(sum(dfdt*InvCutMask*self.grid.RR2))/sum(sum(dist.f*self.grid.RR2))*StepTime)
            energyfracovercut.append(sum(sum(c2*dfdt*InvCutMask*self.grid.RR2))/sum(sum(c2*dist.f*self.grid.RR2))*StepTime)
            dfdts.append(copy.deepcopy(dfdt))
            dist.f=(copy.deepcopy(dist.f)+dfdt*StepTime)*CutMask

            times.append(times[-1]+StepTime)
            if(self.verbose):
                print("Cooling step "+ str(ti)+", t="+str(times[-1])+", dp="+str(particlefracovercut[-1])+", dE="+str(energyfracovercut[-1]))
        ETraj=np.cumprod(1-np.array(energyfracovercut))
        PTraj=np.cumprod(1-np.array(particlefracovercut))

        return fs, dfdts,times,ETraj,PTraj,cutpos
