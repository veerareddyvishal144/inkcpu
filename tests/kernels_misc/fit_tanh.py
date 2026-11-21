import numpy as np
# fit tanh(x) = x*(1 + z*P(z)), z=x^2, |x|<=X, P of degree D (relative error minimisation via iteratively reweighted LSQ ~ minimax)
X=0.625
for D in (4,5,6):
    xs=np.cos(np.linspace(0,np.pi,4000))*0.5+0.5
    xs=xs*X
    xs=xs[xs>1e-4]
    z=xs*xs
    y=(np.tanh(xs)/xs-1)/z
    w=np.ones_like(z)
    for it in range(60):
        A=np.vander(z,D+1,increasing=True)*w[:,None]
        c,*_=np.linalg.lstsq(A,y*w,rcond=None)
        p=np.vander(z,D+1,increasing=True)@c
        err=np.abs(xs*(1+z*p)-np.tanh(xs))/np.tanh(xs)
        w=w*(1+err/err.max())**0.5
    print(D,"maxrel",err.max(),"coeffs",[float(np.float32(v)) for v in c])
