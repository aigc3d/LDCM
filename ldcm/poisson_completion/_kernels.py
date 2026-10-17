"""Elementwise kernels with the reference floating-point operation order."""
import torch
import triton
import triton.language as tl


@triton.jit
def _matvec(P, WEIGHT, MASS, LAMDA, OUT, PRODUCT,
            H:tl.constexpr,W:tl.constexpr,N:tl.constexpr,M:tl.constexpr,BLOCK:tl.constexpr,
            WITH_PRODUCT:tl.constexpr):
    b=tl.program_id(0)
    i=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    ok=i<N
    row=i//W
    col=i%W
    c=tl.load(P+b*N+i,ok,0.)
    l=tl.load(P+b*N+i-1,ok & (col>0),0.)
    r=tl.load(P+b*N+i+1,ok & (col<W-1),0.)
    u=tl.load(P+b*N+i-W,ok & (row>0),0.)
    d=tl.load(P+b*N+i+W,ok & (row<H-1),0.)
    wl=tl.load(WEIGHT+b*M+row*(W-1)+col-1,ok & (col>0),0.)
    wr=tl.load(WEIGHT+b*M+row*(W-1)+col,ok & (col<W-1),0.)
    wu=tl.load(WEIGHT+b*M+H*(W-1)+i-W,ok & (row>0),0.)
    wd=tl.load(WEIGHT+b*M+H*(W-1)+i,ok & (row<H-1),0.)
    left=tl.where(col>0,wl*(c-l),0.)
    right=tl.where(col<W-1,wr*(r-c),0.)
    up=tl.where(row>0,wu*(c-u),0.)
    down=tl.where(row<H-1,wd*(d-c),0.)
    # Preserve zero -> +dx_left -> -dx_right -> +dy_up -> -dy_down.
    value=((0.+left)-right)+up
    value=value-down
    mass=tl.load(MASS+b*N+i,ok,0.)*c
    mass=tl.load(LAMDA+b)*mass
    value=value+mass
    tl.store(OUT+b*N+i,value,ok)
    if WITH_PRODUCT:
        tl.store(PRODUCT+b*N+i,c*value,ok)


def matvec(value,weight,mass,lamda,height,width):
    result=torch.empty_like(value)
    _matvec[(value.shape[0],triton.cdiv(height*width,256))](
        value,weight,mass,lamda,result,result,H=height,W=width,N=height*width,
        M=height*(width-1)+(height-1)*width,BLOCK=256,
        WITH_PRODUCT=False,num_warps=4,enable_fp_fusion=False)
    return result


def matvec_products(value,weight,mass,lamda,height,width):
    """Evaluate Kp and its elementwise contribution to p^T Kp in one pass."""
    result=torch.empty_like(value)
    products=torch.empty_like(value)
    _matvec[(value.shape[0],triton.cdiv(height*width,256))](
        value,weight,mass,lamda,result,products,H=height,W=width,N=height*width,
        M=height*(width-1)+(height-1)*width,BLOCK=256,
        WITH_PRODUCT=True,num_warps=4,enable_fp_fusion=False)
    return result,products


@triton.jit
def _update_x_r(X,R,P,AP,RHO,DENOM,ACTIVE,N:tl.constexpr,BLOCK:tl.constexpr):
    if tl.load(ACTIVE):
        b=tl.program_id(0)
        i=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
        rho=tl.load(RHO+b)
        den=tl.load(DENOM+b)
        alpha=tl.div_rn(rho,tl.where(den==0,1e-8,den))
        x=tl.load(X+b*N+i,i<N,0.)
        p=tl.load(P+b*N+i,i<N,0.)
        r=tl.load(R+b*N+i,i<N,0.)
        ap=tl.load(AP+b*N+i,i<N,0.)
        tl.store(X+b*N+i,x+alpha*p,i<N)
        tl.store(R+b*N+i,r-alpha*ap,i<N)


@triton.jit
def _control(NORM,LAST,THRESHOLD,RHO,NEW_RHO,BETA,ACTIVE,STEPS,STALLED,CAP,
             BATCH:tl.constexpr,BLOCK:tl.constexpr):
    if tl.load(ACTIVE):
        i=tl.arange(0,BLOCK)
        norm=tl.load(NORM+i,i<BATCH,0.)
        last=tl.load(LAST+i,i<BATCH,0.)
        threshold=tl.load(THRESHOLD+i,i<BATCH,0.)
        converged=tl.sum(((i<BATCH) & ~(norm<=threshold)).to(tl.int32),0)==0
        improvement=tl.div_rn(last-norm,last+1e-8)
        any_nan=tl.sum(((i<BATCH) & (improvement!=improvement)).to(tl.int32),0)>0
        best=tl.max(tl.where(i<BATCH,improvement,-float('inf')),0)
        stalled=tl.where((best<1e-4) & ~any_nan,tl.load(STALLED)+1,0)
        step=tl.load(STEPS)+1
        active=(~converged) & (stalled<10) & (step<tl.load(CAP))
        rho=tl.load(RHO+i,i<BATCH,0.)
        new=tl.load(NEW_RHO+i,i<BATCH,0.)
        beta=tl.div_rn(new,tl.where(rho==0,1e-8,rho))
        tl.store(BETA+i,beta,i<BATCH)
        tl.store(RHO+i,new,i<BATCH)
        tl.store(LAST+i,norm,i<BATCH)
        tl.store(ACTIVE,active)
        tl.store(STEPS,step)
        tl.store(STALLED,stalled)


@triton.jit
def _update_p(P,R,BETA,ACTIVE,N:tl.constexpr,BLOCK:tl.constexpr):
    if tl.load(ACTIVE):
        b=tl.program_id(0)
        i=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
        beta=tl.load(BETA+b)
        p=tl.load(P+b*N+i,i<N,0.)
        r=tl.load(R+b*N+i,i<N,0.)
        tl.store(P+b*N+i,r+beta*p,i<N)


def update_x_r(x,r,p,ap,rho,denominator,active):
    _update_x_r[(x.shape[0],triton.cdiv(x.shape[1],256))](
        x,r,p,ap,rho,denominator,active,N=x.shape[1],BLOCK=256,
        num_warps=4,enable_fp_fusion=False)


def control(norm,last,threshold,rho,new_rho,beta,active,steps,stalled,cap):
    batch=norm.numel()
    _control[(1,)](norm,last,threshold,rho,new_rho,beta,active,steps,stalled,cap,
                  BATCH=batch,BLOCK=triton.next_power_of_2(batch),num_warps=4,
                  enable_fp_fusion=False)


def update_p(p,r,beta,active):
    _update_p[(p.shape[0],triton.cdiv(p.shape[1],256))](
        p,r,beta,active,N=p.shape[1],BLOCK=256,num_warps=4,enable_fp_fusion=False)
