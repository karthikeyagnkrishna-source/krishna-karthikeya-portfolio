"""Unrounded float64 strict scoring; undefined percentages remain NaN."""
import numpy as np
from contracts import ToleranceConfig

def _finite(actual, predicted):
    a,p = np.broadcast_arrays(np.asarray(actual,dtype=np.float64),np.asarray(predicted,dtype=np.float64))
    if not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite physical value cannot be scored')
    return a,p

def score_scalar(actual,predicted,kind,cfg=ToleranceConfig()):
    a,p = _finite(actual,predicted)
    if kind == 'angle': a,p = np.rad2deg(a),np.rad2deg(p)
    e = abs(p-a)
    z = abs(a) <= cfg.zero_threshold
    tolerance = np.full_like(a,cfg.contact) if kind == 'contact' else np.where(z,getattr(cfg,kind),cfg.relative*abs(a))
    percent = np.full_like(a,np.nan)
    np.divide(e*100,abs(a),out=percent,where=~z)
    return dict(actual=a,predicted=p,error=e,zero=z,tolerance=tolerance,percent=percent,
                ratio=e/tolerance,**{'pass':e < tolerance})

def score_vector(actual,predicted,kind,cfg=ToleranceConfig()):
    a,p = _finite(actual,predicted)
    magnitude=np.linalg.norm(a,axis=-1)
    e=np.linalg.norm(p-a,axis=-1)
    z=magnitude<=cfg.zero_threshold
    tolerance=np.where(z,cfg.force if kind=='force' else 0.,cfg.relative*magnitude)
    percent=np.full_like(e,np.nan)
    np.divide(e*100,magnitude,out=percent,where=~z)
    ratio=np.full_like(e,np.inf)
    np.divide(e,tolerance,out=ratio,where=tolerance>0)
    ratio=np.where(z & (e==0),0,ratio)
    passed=np.where(z & (kind=='foot'),e==0,e<tolerance)
    return dict(actual=magnitude,predicted=np.linalg.norm(p,axis=-1),error=e,zero=z,tolerance=tolerance,
                percent=percent,ratio=ratio,**{'pass':passed})

def strict_summary(rows):
    e=np.asarray(rows['error']).ravel(); p=np.asarray(rows['percent']).ravel()
    passed=np.asarray(rows['pass']).ravel(); zero=np.asarray(rows['zero']).ravel()
    valid=p[np.isfinite(p)]
    return dict(points=len(e),failures=int((~passed).sum()),zero_points=int(zero.sum()),
                zero_failures=int((zero & ~passed).sum()),pass_rate=float(passed.mean()) if len(e) else None,
                MAE=float(e.mean()) if len(e) else None,mean_percent=float(valid.mean()) if len(valid) else None,
                p95_error=float(np.percentile(e,95)) if len(e) else None,
                p99_error=float(np.percentile(e,99)) if len(e) else None,max_error=float(e.max()) if len(e) else None,
                max_percent=float(valid.max()) if len(valid) else None)
