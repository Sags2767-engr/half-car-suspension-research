"""
real.py  -  Realistic chain: first-order actuator lag + force limit, noisy sensors, steady-state
Kalman filter, 0.5 Hz washout on estimated body velocity, ISO 2631-1:1997 Annex A Wk weighting.
"""
import numpy as np
from scipy.linalg import expm, solve_discrete_are
from hc import *
# ---------------- ISO 2631-1:1997 Wk weighting (Annex A transfer function)
# Frequency response is evaluated as a complex transfer function; wrms uses
# its magnitude, which is sufficient for the total RMS of a Fourier-weighted signal.
_F=np.array([0.5,0.63,0.8,1,1.25,1.6,2,2.5,3.15,4,5,6.3,8,10,12.5,16,20,25,31.5,40,50,63,80,100])
_W=np.array([0.418,0.459,0.477,0.482,0.484,0.494,0.531,0.631,0.804,0.967,1.039,1.054,1.036,0.988,0.902,0.768,0.636,0.513,0.405,0.314,0.246,0.186,0.132,0.0887])

def wk_gain(f):
    """Magnitude of the ISO 2631-1:1997 Annex A Wk transfer function.

    Parameters are f1=0.4 Hz, f2=100 Hz, f3=f4=12.5 Hz, Q4=0.63,
    f5=2.37 Hz, Q5=0.91, f6=3.35 Hz, Q6=0.91.
    """
    f=np.asarray(f,dtype=float)
    s=1j*2*np.pi*f
    w1,w2,w3,w4,w5,w6=2*np.pi*np.array([0.4,100.0,12.5,12.5,2.37,3.35])
    q4,q5,q6=0.63,0.91,0.91
    Hh=s**2/(s**2+np.sqrt(2)*w1*s+w1**2)
    Hl=w2**2/(s**2+np.sqrt(2)*w2*s+w2**2)
    Ht=(1+s/w3)/(1+s/(q4*w4)+(s/w4)**2)
    Hs=((1+s/(q5*w5)+(s/w5)**2)/
        (1+s/(q6*w6)+(s/w6)**2))*(w5/w6)**2
    return np.abs(Hh*Hl*Ht*Hs)

def wrms(x,dt,skip=0):
    x=x[skip:]-np.mean(x[skip:]);X=np.fft.rfft(x);f=np.fft.rfftfreq(len(x),dt)
    return rms(np.fft.irfft(X*wk_gain(f),len(x)))
# ---------------- realistic chain: actuator lag + saturation, noisy sensors, Kalman filter
SENS=dict(s=0.5e-3,ab=0.05,aw=0.10)       # std of suspension-travel, body-acc, wheel-acc sensors
ACT=dict(fbw=20.0,fmax=2000.0)            # actuator bandwidth (Hz), force limit (N)
def design(p,dt=0.001,v=20.0,Gd=64e-6,qscale=1.0,sens=SENS,act=ACT):
    Ad,Bd,E,_,_,_,_=defmodel(p);tau=1/(2*np.pi*act['fbw'])
    n=10
    Aa=np.zeros((n,n));Aa[:8,:8]=Ad;Aa[:8,8:]=Bd;Aa[8:,8:]=-np.eye(2)/tau
    Ba=np.zeros((n,4));Ba[8:,:2]=np.eye(2)/tau;Ba[:8,2:]=E
    M=np.zeros((n+4,n+4));M[:n,:n]=Aa*dt;M[:n,n:]=Ba*dt;Mx=expm(M);Phi=Mx[:n,:n];Gam=Mx[:n,n:]
    C=np.zeros((6,n));C[0,0]=1;C[1,1]=1
    C[2,:8]=Ad[4];C[2,8:]=Bd[4];C[3,:8]=Ad[5];C[3,8:]=Bd[5]
    C[4,:8]=Ad[6];C[4,8:]=Bd[6];C[5,:8]=Ad[7];C[5,8:]=Bd[7]
    Rn=np.diag([sens['s']**2]*2+[sens['ab']**2]*2+[sens['aw']**2]*2)
    q=4*np.pi**2*Gd*0.01*v/(4*np.pi)*qscale     # two-sided white-noise intensity of road velocity (ISO 8608)
    Qd=np.zeros((n,n));Qd[:8,:8]=E@E.T*q*dt;Qd+=np.eye(n)*1e-14
    Pk=solve_discrete_are(Phi.T,C.T,Qd,Rn);Lk=Phi@Pk@C.T@np.linalg.inv(C@Pk@C.T+Rn)
    return dict(Phi=Phi,Gam=Gam,C=C,L=Lk,dt=dt,tau=tau,Ad=Ad,Bd=Bd,E=E,act=act)
WASH=dict(idx=[4,5],fc=0.5)   # high-pass on the estimated absolute body velocities
def run(p,K,road,t,D,mode='full',seed=0,sens=SENS,noise=True,wash=WASH):
    """mode: 'act' = state feedback + actuator limits ; 'full' = actuator limits + Kalman filter with noisy sensors"""
    Phi,Gam,C,Lk=D['Phi'],D['Gam'],D['C'],D['L'];Ad,Bd,E=D['Ad'],D['Bd'],D['E'];fmax=D['act']['fmax']
    # plant may differ from the design model (parameter errors): rebuild plant matrices
    if p is not None and D.get('plant') is not None:
        Phi_p,Gam_p,C_p=D['plant']['Phi'],D['plant']['Gam'],D['plant']['C']
    else: Phi_p,Gam_p,C_p=Phi,Gam,C
    wf,wr,df,dr=road;N=len(t);x=np.zeros(10);xh=np.zeros(10)
    rng=np.random.default_rng(seed);sg=np.array([sens['s']]*2+[sens['ab']]*2+[sens['aw']]*2)
    Xs=np.empty((N,10));Xh=np.empty((N,10));Uc=np.empty((N,2))
    idx=wash['idx'] if wash else [];a=np.exp(-2*np.pi*wash['fc']*D['dt']) if wash else 0.0;z=np.zeros(len(idx))
    for k in range(N):
        if mode=='act': ctrl=-K@x[:8]
        else:
            xe=xh[:8].copy()
            if wash: xe[idx]-=z;z=a*z+(1-a)*xh[idx]
            ctrl=-K@xe
        uc=np.clip(ctrl,-fmax,fmax);Uc[k]=uc;Xs[k]=x;Xh[k]=xh
        inp=np.array([uc[0],uc[1],df[k],dr[k]])
        if mode=='full':
            y=C_p@x+(rng.standard_normal(6)*sg if noise else 0)
            xh=Phi@xh+Gam@np.array([uc[0],uc[1],0,0])+Lk@(y-C@xh)
        x=Phi_p@x+Gam_p@inp
    return Xs,Xh,Uc
def outputs(p,Xs,Xh,road,t):
    Ad,Bd,E,_,_,_,_=defmodel(p);wf,wr,df,dr=road;s=Xs[:,:8];ua=Xs[:,8:]
    U=np.column_stack([df,dr]);xd=s@Ad.T+ua@Bd.T+U@E.T;L=p['Lf']+p['Lr']
    abf,abr=xd[:,4],xd[:,5];ab=(p['Lr']*abf+p['Lf']*abr)/L;ath=(abf-abr)/L
    xbf=s[:,0]+s[:,2]+wf;xbr=s[:,1]+s[:,3]+wr;xb=(p['Lr']*xbf+p['Lf']*xbr)/L;th=(xbf-xbr)/L
    return dict(t=t,X=s,u=ua,ab=ab,ath=ath,xb=xb,th=th,sf=s[:,0],sr=s[:,1],tf=s[:,2],tr=s[:,3])