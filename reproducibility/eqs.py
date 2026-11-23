"""
eqs.py  -  Renders the 17 equations as PNG (only used for the PDF preview of the manuscript).
"""
import os
os.makedirs('eqs', exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset']='stix'
E={
1:r'F_i = k_{bi}\,(x_{wi}-x_{bi}) + c_{bi}\,(\dot{x}_{wi}-\dot{x}_{bi}) + u_i',
2:r'm_b\,\ddot{x}_b = F_f + F_r',
3:r'I_b\,\ddot{\theta} = L_f F_f - L_r F_r',
4:r'm_{wi}\,\ddot{x}_{wi} = -F_i - k_{wi}\,(x_{wi}-w_i)',
5:r'x_{bf} = x_b + L_f\,\theta,\quad x_{br} = x_b - L_r\,\theta',
6:r'x_b = \frac{L_r\,x_{bf} + L_f\,x_{br}}{L},\quad \theta = \frac{x_{bf}-x_{br}}{L}',
7:r'\dot{x} = A\,x + B\,u + G\,w',
8:r'\dot{s} = \tilde{A}\,s + \tilde{B}\,u + \tilde{E}\,\dot{w}',
9:r'J=\int_0^{\infty}\left[\alpha\,(\ddot{x}_b^{2}+\ddot{\theta}^{2})+\gamma\,\frac{s_f^{2}+s_r^{2}}{s_0^{2}}+\beta\,\frac{t_f^{2}+t_r^{2}}{t_0^{2}}+\rho\,\frac{u_f^{2}+u_r^{2}}{u_0^{2}}\right]dt',
10:r'u=-K\,s,\quad K=R^{-1}\,(\tilde{B}^{T}P+N^{T})',
11:r'u_i=-c_{sky}\,\dot{x}_{bi}',
12:r'\tau\,\dot{u}_{a,i} = \mathrm{sat}_{F_{max}}(u_{c,i}) - u_{a,i},\quad \tau = \frac{1}{2\pi f_{bw}}',
13:r'\hat{z}_{k+1} = \Phi\,\hat{z}_k + \Gamma\,u_{c,k} + L\,(y_k - C\,\hat{z}_k)',
14:r'w(t)=\frac{h}{2}\left[1-\cos\left(\frac{2\pi v t}{\ell}\right)\right],\quad 0\leq t\leq \ell/v',
15:r'G_d(n)=G_d(n_0)\left(\frac{n}{n_0}\right)^{-2}',
16:r'w(x)=\sum_{i=1}^{300}a_i\sin(2\pi n_i x+\varphi_i),\quad a_i=\sqrt{2\,G_d(n_i)\,\Delta n}',
17:r'\mathrm{DTL}_i=\frac{k_{wi}\,|x_{wi}-w_i|}{F_{static,i}}',
}
for k,v in E.items():
    fig=plt.figure(figsize=(6,1));fig.text(0,0.5,'$'+v+'$',fontsize=13,va='center')
    fig.savefig(f'eqs/eq{k}.png',dpi=300,bbox_inches='tight',pad_inches=0.04,facecolor='white');plt.close(fig)
print('eqs ok')