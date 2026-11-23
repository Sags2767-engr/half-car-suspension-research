"""
hc.py  -  Half-car model, LQR design, road inputs, continuous-time simulation and metrics.
State convention: x = [xbf xbr xwf xwr vbf vbr vwf vwr] (Eq. 7); LQR uses deflection coordinates
s = [sf sr tf tr vbf vbr vwf vwr] (Eq. 8). Parameters NOM = Table 2 of the paper.
"""
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import lsim, StateSpace
NOM=dict(mb=1794.4,Ib=3443.05,mwf=87.15,mwr=140.14,kbf=66824.2,kbr=18615.0,
         kwf=101115.0,kwr=101115.0,cbf=1190.0,cbr=1000.0,Lf=1.271,Lr=1.713)
G0=9.81
def absmodel(p):
    mb,Ib,mwf,mwr=p['mb'],p['Ib'],p['mwf'],p['mwr']
    kbf,kbr,kwf,kwr,cbf,cbr,Lf,Lr=[p[k] for k in ('kbf','kbr','kwf','kwr','cbf','cbr','Lf','Lr')]
    Mi=np.array([[1/mb+Lf**2/Ib,1/mb-Lf*Lr/Ib],[1/mb-Lf*Lr/Ib,1/mb+Lr**2/Ib]])
    A=np.zeros((8,8));A[0:4,4:8]=np.eye(4)
    Ax=np.zeros((2,4));Ax[0,0]=-kbf;Ax[0,2]=kbf;Ax[1,1]=-kbr;Ax[1,3]=kbr
    Av=np.zeros((2,4));Av[0,0]=-cbf;Av[0,2]=cbf;Av[1,1]=-cbr;Av[1,3]=cbr
    A[4:6,0:4]=Mi@Ax;A[4:6,4:8]=Mi@Av
    A[6,0]=kbf/mwf;A[6,2]=-(kbf+kwf)/mwf;A[6,4]=cbf/mwf;A[6,6]=-cbf/mwf
    A[7,1]=kbr/mwr;A[7,3]=-(kbr+kwr)/mwr;A[7,5]=cbr/mwr;A[7,7]=-cbr/mwr
    Gm=np.zeros((8,2));Gm[6,0]=kwf/mwf;Gm[7,1]=kwr/mwr
    B=np.zeros((8,2));B[4:6,:]=Mi;B[6,0]=-1/mwf;B[7,1]=-1/mwr
    return A,B,Gm
def defmodel(p):
    A,B,Gm=absmodel(p)
    T=np.zeros((8,8));T[0,0]=1;T[0,2]=-1;T[1,1]=1;T[1,3]=-1;T[2,2]=1;T[3,3]=1;T[4:,4:]=np.eye(4)
    S=np.zeros((8,2));S[2,0]=-1;S[3,1]=-1
    Ti=np.linalg.inv(T)
    Ad=T@A@Ti
    resid=np.abs(-Ad@S+T@Gm).max()
    E=S.copy();Bd=T@B
    return Ad,Bd,E,resid,A,B,Gm
def lqr(p,alpha=1.0,beta=1.0,gs=1.0,ru=1.0,s0=0.1,t0=0.02,u0=1000.0):
    Ad,Bd,E,_,_,_,_=defmodel(p)
    Lf,Lr=p['Lf'],p['Lr'];L=Lf+Lr
    Mrow=np.zeros((2,8));Mrow[0]=Ad[4];Mrow[1]=Ad[5]
    Db=Bd[4:6,:]
    Cb=np.array([[Lr/L,Lf/L],[1/L,-1/L]])
    C1=Cb@Mrow;D1=Cb@Db
    wts=[];Cs=[];Ds=[]
    Cs.append(C1);Ds.append(D1);wts+=[alpha,alpha]
    for idx,sc,w in ((0,s0,gs),(1,s0,gs),(2,t0,beta),(3,t0,beta)):
        c=np.zeros((1,8));c[0,idx]=1;Cs.append(c);Ds.append(np.zeros((1,2)));wts.append(w/sc**2)
    C=np.vstack(Cs);D=np.vstack(Ds);W=np.diag(wts)
    Q=C.T@W@C;N=C.T@W@D;R=D.T@W@D+ru*np.eye(2)/u0**2
    P=solve_continuous_are(Ad,Bd,Q,R,s=N)
    return np.linalg.solve(R,Bd.T@P+N.T)
def road_bump(h,ell,v,tdel,t):
    T=ell/v
    def f(tt):
        m=(tt>=0)&(tt<=T)
        w=np.where(m,h/2*(1-np.cos(2*np.pi*tt/T)),0.0)
        dw=np.where(m,h/2*(2*np.pi/T)*np.sin(2*np.pi*tt/T),0.0)
        return w,dw
    wf,df=f(t);wr,dr=f(t-tdel);return wf,wr,df,dr
def road_iso(Gd,v,tdel,t,rng,n1=0.011,n2=2.83,N=300):
    dn=(n2-n1)/N;n=n1+dn*(np.arange(N)+0.5)
    a=np.sqrt(2*Gd*(0.1**2)/n**2*dn)
    ph=rng.uniform(0,2*np.pi,N)
    def f(tt):
        arg=2*np.pi*np.outer(n*v,tt)+ph[:,None]
        return (a[:,None]*np.sin(arg)).sum(0),(a[:,None]*2*np.pi*(n*v)[:,None]*np.cos(arg)).sum(0)
    wf,df=f(t);wr,dr=f(t-tdel);return wf,wr,df,dr
def simulate(p,ctrl,road,t,x0=None):
    Ad,Bd,E,_,_,_,_=defmodel(p)
    wf,wr,df,dr=road
    if ctrl is None: Kc=np.zeros((2,8))
    elif ctrl[0]=='lqr': Kc=ctrl[1]
    else:
        Kc=np.zeros((2,8));Kc[0,4]=ctrl[1];Kc[1,5]=ctrl[2]
    Acl=Ad-Bd@Kc
    sys=StateSpace(Acl,E,np.eye(8),np.zeros((8,2)))
    U=np.column_stack([df,dr])
    _,_,X=lsim(sys,U,t,X0=np.zeros(8) if x0 is None else x0,interp=True)
    u=-(X@Kc.T)
    xd=X@Ad.T+u@Bd.T+U@E.T
    L=p['Lf']+p['Lr']
    abf,abr=xd[:,4],xd[:,5]
    ab=(p['Lr']*abf+p['Lf']*abr)/L; ath=(abf-abr)/L
    xbf=X[:,0]+X[:,2]+wf; xbr=X[:,1]+X[:,3]+wr
    xb=(p['Lr']*xbf+p['Lf']*xbr)/L; th=(xbf-xbr)/L
    return dict(t=t,X=X,u=u,ab=ab,ath=ath,xb=xb,th=th,sf=X[:,0],sr=X[:,1],tf=X[:,2],tr=X[:,3],
                xwf=X[:,2]+wf,xwr=X[:,3]+wr,wf=wf,wr=wr,abf=abf,abr=abr)
def rms(x): return float(np.sqrt(np.mean(np.square(x))))
def settle(t,y,tol=0.02,floor=1e-4):
    pk=np.max(np.abs(y)); thr=max(tol*pk,floor)
    idx=np.where(np.abs(y)>thr)[0]
    return float(t[idx[-1]]) if len(idx) else 0.0
def metrics(p,r,bump=False):
    Ft=p['kwf']*r['tf'];Rt=p['kwr']*r['tr']
    Fs=(p['mb']*G0*p['Lr']/(p['Lf']+p['Lr'])+p['mwf']*G0, p['mb']*G0*p['Lf']/(p['Lf']+p['Lr'])+p['mwr']*G0)
    m=dict(ab_rms=rms(r['ab']),ab_pk=float(np.max(np.abs(r['ab']))),ath_rms=rms(r['ath']),th_pk=float(np.max(np.abs(r['th']))),
      xb_pk=float(np.max(np.abs(r['xb']))),s_pk=float(max(np.max(np.abs(r['sf'])),np.max(np.abs(r['sr'])))),
      s_rms=float(max(rms(r['sf']),rms(r['sr']))),
      dtl_rms=float(max(rms(Ft)/Fs[0],rms(Rt)/Fs[1])),dtl_pk=float(max(np.max(np.abs(Ft))/Fs[0],np.max(np.abs(Rt))/Fs[1])),
      u_pk=float(np.max(np.abs(r['u']))),u_rms=float(max(rms(r['u'][:,0]),rms(r['u'][:,1]))))
    if bump:
        m['ts_xb']=settle(r['t'],r['xb']);m['ts_th']=settle(r['t'],r['th'])
    return m