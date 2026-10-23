# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1
import time, os, random, numpy as np
GLOBAL_START=time.monotonic()
SEED=42
random.seed(SEED);np.random.seed(SEED)
# Import Colab's existing CUDA-enabled torch BEFORE any package installation.
# Third-party device backend autoloading is unnecessary for the Colab GPU.
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
try:
    import torch
except RuntimeError as exc:
    if 'TORCH_LIBRARY' in str(exc) or 'namespace prims' in str(exc):
        raise RuntimeError('PyTorch was partly imported in this Colab session. Use Runtime → Restart session, then Run all. If a fresh restart also fails, Runtime → Disconnect and delete runtime, reconnect with GPU, then Run all. Re-running this cell alone cannot repair the operator registry.') from exc
    raise
assert torch.cuda.is_available(), 'Choose Runtime → Change runtime type → T4 GPU (or faster), then Restart session and Run all.'
print('Started total setup + data + training timer; torch',torch.__version__,'GPU',torch.cuda.get_device_name(0))

# %% notebook cell 2
import importlib.util, subprocess, sys
packages={'transformers':'transformers>=4.44,<5',
          'cv2':'opencv-python-headless>=4.8,<5',
          'scipy':'scipy>=1.11,<2',
          'pandas':'pandas',
          'matplotlib':'matplotlib'}
missing=[spec for module,spec in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    print('Installing only missing packages:',missing)
    subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
else:
    print('Required libraries already installed; PyTorch and its CUDA packages were left untouched.')

# %% notebook cell 4
import gc
assert torch.cuda.is_available(), 'Choose Runtime → Change runtime type → T4 GPU (or faster); CPU will not meet the time target.'
device='cuda'
torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32=True
CACHE_ROOT='/content/feelwm_td2_short_cache'
TOTAL_MINUTES=30
CLIP_SECONDS=8.0
MAX_FRAMES=75
MAX_UPDATES=16
print('GPU:',torch.cuda.get_device_name(0),'cache:',CACHE_ROOT)

# %% notebook cell 6
"""Feel-WM on the paper's TartanDrive 2.0 ATV data. Notebook inlines this file."""
import io, os, json, re, time, tarfile, urllib.request, urllib.error
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

BUCKET='https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartandrive2/'
BASE='kitti/all_topics/'
RUNS={
 'train': ['2023-11-02-16-00-43_down_meadows','2023-11-14-15-02-21_figure_8'],
 'validation':['2023-11-14-15-22-01_figure_8'],
 'test':['2023-11-14-15-11-49_rough_rider_turnpike_loop'],
}
PAPER_TABLE={
 ('Gazebo','Park'):[(3.83,.50),(3.74,.49),(3.71,.48)],
 ('Gazebo','Forest'):[(4.08,.52),(4.07,.74),(4.00,.52)],
 ('Gazebo','Hill'):[(3.75,.52),(3.62,.45),(3.53,.41)],
 ('Isaac Sim','Smooth'):[(5.54,.45),(5.68,.45),(5.62,.40)],
 ('Isaac Sim','Rough'):[(6.46,.73),(6.53,.77),(6.51,.65)],
}

class Deadline:
 def __init__(self,minutes=28,start=None):
  self.start=time.monotonic() if start is None else start
  self.until=self.start+minutes*60;self.phases=[];self.bytes=0
 def left(self):return self.until-time.monotonic()
 def require(self,reserve=60):
  if self.left()<reserve:raise TimeoutError('Time budget reached; rerun cached preparation or increase TOTAL_MINUTES. No unmeasured result is emitted.')
 def log(self,name,start,**fields):
  row={'phase':name,'seconds':round(time.monotonic()-start,2),**fields};self.phases.append(row);print(row,flush=True)

def hold(t,x,q):
 j=np.searchsorted(t,np.asarray(q),side='right')-1
 if np.any(j<0):raise ValueError('Causal stream not available before query')
 return np.asarray(x)[j]

def yaw_from_quat(q):return Rotation.from_quat(q).as_euler('xyz')[:,2]
def ride(t,vertical_acc,yaw):
 t=np.asarray(t);y=np.unwrap(yaw)
 return {'roughness_m_s2':float(np.std(vertical_acc,ddof=0)),
         'yaw_vibration_rad_s':float(np.mean(np.abs(np.diff(y)/np.diff(t))))}

class RemoteFile(io.RawIOBase):
 """Bounded, validated HTTP byte-range reader. Caches completed 4 MiB blocks."""
 def __init__(self,key,cache,deadline,block=4*1024**2):
  self.url=BUCKET+key;self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
  self.deadline=deadline;self.block=block;self.pos=0;self.mem={}
  p=self.cache/'size.txt'
  if p.exists():self.size=int(p.read_text())
  else:
   b,total=self.get_range(0,511);self.size=total;p.write_text(str(total))
 def get_range(self,a,b):
  error=None
  for _ in range(3):
   self.deadline.require(100)
   try:
    # Change URL with the range so intermediary caches cannot mix ranges.
    u=self.url+f'?byte_begin={a}&byte_end={b}'
    request=urllib.request.Request(u,headers={'Range':f'bytes={a}-{b}','Accept-Encoding':'identity'})
    with urllib.request.urlopen(request,timeout=min(65,max(15,self.deadline.left()-100))) as r:
     m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
     if r.status!=206 or m is None or int(m[1])!=a:raise IOError('Server did not honor byte range')
     need=int(m[2])-a+1
     if need>b-a+1:raise IOError('Unexpected oversized HTTP response')
     payload=r.read(need+1)
     if len(payload)!=need:raise IOError('Truncated/oversized HTTP response')
     self.deadline.bytes+=need;return payload,int(m[3])
   except (urllib.error.URLError,TimeoutError,IOError) as exc:
    error=exc;time.sleep(1)
  raise RuntimeError(f'Cannot read official TartanDrive 2 object {self.url}: {error}')
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=int(offset if whence==0 else self.pos+offset if whence==1 else self.size+offset)
  if self.pos<0:raise ValueError('Negative seek')
  return self.pos
 def read(self,n=-1):
  n=max(0,min(self.size-self.pos,n if n>=0 else self.size-self.pos));out=[]
  while n:
   bid=self.pos//self.block;start=bid*self.block
   if bid in self.mem:buf=self.mem[bid]
   else:
    path=self.cache/f'{bid:06d}.bin'
    if path.exists():buf=path.read_bytes()
    else:
     buf,_=self.get_range(start,min(start+self.block,self.size)-1)
     temp=path.with_suffix('.tmp');temp.write_bytes(buf);temp.replace(path)
    self.mem[bid]=buf
    if len(self.mem)>2:self.mem.pop(next(iter(self.mem)))
   take=min(n,len(buf)-self.pos%self.block)
   if take<=0:raise IOError('Corrupted range cache')
   out.append(buf[self.pos%self.block:self.pos%self.block+take]);self.pos+=take;n-=take
  return b''.join(out)

def official_file(key,folder,deadline):
 target=Path(folder)/key
 if target.exists():return target
 target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix(target.suffix+'.part')
 deadline.require(100)
 with urllib.request.urlopen(BUCKET+key,timeout=min(75,max(15,deadline.left()-100))) as r,tmp.open('wb') as f:
  while chunk:=r.read(1024**2):
   deadline.require(100);f.write(chunk);deadline.bytes+=len(chunk)
 tmp.replace(target);return target

def numeric(run,channel,stem,root,deadline):
 key=f'{BASE}{run}/{channel}/{stem}.npy'
 x=np.load(official_file(key,root,deadline),allow_pickle=False)
 t=np.loadtxt(official_file(f'{BASE}{run}/{channel}/timestamps.txt',root,deadline))
 if t.ndim>1:t=t[:,0]
 if len(t)!=len(x) or len(x)<100:raise RuntimeError(f'Invalid {run}/{channel}: {x.shape}, {t.shape}')
 if np.any(np.diff(t)<=0):
  # The official odometry text contains occasional duplicate timestamps.
  order=np.argsort(t,kind='stable');t=t[order];x=x[order]
  keep=np.r_[t[1:]!=t[:-1],True]
  print(f'{run}/{channel}: deduplicated {len(t)-int(keep.sum())} repeated timestamps')
  t=t[keep];x=x[keep]
 # GPS coordinates exceed four million meters; float32 would quantize positions.
 return t.astype(np.float64),x.astype(np.float64)

def inspect_numeric(run,root,deadline):
 streams={}
 for channel,stem in [('gps_odom','odometry'),('multisense_imu','imu'),('wheel_rpm','encoder'),
                       ('cmd','twist'),('shock_pos','shock')]:
  streams[channel]=numeric(run,channel,stem,root,deadline)
  print(run,channel,streams[channel][1].shape,flush=True)
 return streams

def tar_prefix(run,kind,max_frames,cache,deadline):
 """Read only a short initial clip from the official uncompressed per-modality TAR."""
 key=f'{BASE}{run}/{kind}.tar';rf=RemoteFile(key,Path(cache)/run/kind,deadline)
 members=[]
 with tarfile.open(fileobj=rf,mode='r|') as tar:
  for member in tar:
   if not member.isfile():continue
   if member.size>12*1024**2:raise RuntimeError('Unexpected large image member')
   data=tar.extractfile(member).read();members.append((member.name,data))
   if len(members)>=max_frames:break
 if len(members)<max_frames:raise RuntimeError(f'{kind} ended before {max_frames} images')
 return members

def image_bytes(payload):
 import cv2
 x=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_UNCHANGED)
 if x is None:
  try:x=np.load(io.BytesIO(payload),allow_pickle=False)
  except Exception as e:raise RuntimeError('Unsupported official KITTI image format') from e
 return x

def robust_imu_vertical(imu,odom_q,odom_t,imu_t,calibration_end):
 """Estimate fixed sensor-to-body rotation from calibration-only low-motion samples.

 The IMU axes are not body axes (the provided example has gravity in sensor x).
 A gravity direction is estimated on the first calibration interval. Roughness
 uses the *signed projection*, whose standard deviation is independent of sign.
 If attitude varies during calibration this estimate must be checked externally.
 """
 a=imu[:,-3:]
 # Principal direction of gravity in sensor coordinates estimated before scoring.
 idx=(imu_t>=imu_t[0]+.3)&(imu_t<=imu_t[0]+calibration_end)
 grav=np.median(a[idx],axis=0);grav/=np.linalg.norm(grav)
 if not np.isfinite(grav).all():raise RuntimeError('Cannot calibrate accelerometer')
 # Direction in IMU corresponding to local vertical is assumed fixed over this short clip.
 return a@grav, {'imu_gravity_axis':grav.tolist(),'calibration_seconds':calibration_end,
                 'vertical_method':'projection onto initial-calibration gravity axis; sign irrelevant to std'}

def make_physics(run,root,deadline,seconds=11.5,calibration=2.5):
 start=time.monotonic();s=inspect_numeric(run,root,deadline)
 ot,odom=s['gps_odom'];it,imu=s['multisense_imu'];wt,wheel=s['wheel_rpm'];ct,cmd=s['cmd'];st,shock=s['shock_pos']
 # The official converter concatenates pose[7], twist[6] and covariance.
 if odom.shape[1]<13 or imu.shape[1]<6 or wheel.shape[1]<4:
  raise RuntimeError(f'Official numeric schema differs: odom {odom.shape}, imu {imu.shape}, wheel {wheel.shape}')
 base=max(ot[0],it[0],wt[0],ct[0],st[0]);last=min(ot[-1],it[-1],wt[-1],ct[-1],st[-1]);
 if last-base<seconds:raise RuntimeError(f'Run {run} shorter than selected {seconds}s clip')
 # Verify quaternion slice empirically against unit norm; no guessed pose fields.
 candidate=[(3,7),(6,10)]
 qslice=next(((a,b) for a,b in candidate if b<=odom.shape[1] and
              np.mean(np.abs(np.linalg.norm(odom[:100,b-4:b],axis=1)-1))<.03),None)
 if qslice is None:raise RuntimeError('Could not verify odometry quaternion layout')
 q=odom[:,qslice[0]:qslice[1]];xyz=odom[:,:3];ot=ot-base;it=it-base;wt=wt-base;ct=ct-base;st=st-base
 imu_vertical,audit=robust_imu_vertical(imu,q,ot,it,calibration)
 grid=np.arange(.0,seconds,.01);grid=grid[grid>=max(ot[0],it[0],wt[0],ct[0],st[0])]
 if grid[0]>.1:raise RuntimeError('Initial sensor overlap too late')
 imu100=hold(it,imu,grid);acc=hold(it,imu_vertical,grid);wheel100=hold(wt,wheel,grid)[:,:4]
 odom100=hold(ot,odom,grid);p=hold(ot,xyz,grid);quat=hold(ot,q,grid)
 yaw=np.unwrap(yaw_from_quat(quat));vel=np.linalg.norm(np.gradient(p[:,:2],.01,axis=0),axis=1)
 # Encoder units must be checked; infer a positive train-only speed scale later.
 # Other physical targets: 0.5s centered roughness and roll/pitch instability.
 rough=np.array([np.std(acc[max(0,i-25):min(len(acc),i+26)]) for i in range(len(grid))])
 gravity_axis=np.asarray(audit['imu_gravity_axis']);gyro_horizontal=imu100[:,:3]-(imu100[:,:3]@gravity_axis)[:,None]*gravity_axis
 angular=np.sqrt(np.array([np.var(gyro_horizontal[max(0,i-25):min(len(acc),i+26)],axis=0).sum()
                           for i in range(len(grid))]))
 pitch=Rotation.from_quat(quat).as_euler('xyz')[:,1]
 slope=vel*np.sin(pitch)
 # Keep 4 actual wheels as individual signals. They are NOT leg joint/force channels.
 state=np.c_[imu100,wheel100,hold(st,shock,grid)[:,:4],odom100[:,7:13],np.gradient(yaw,.01),vel].astype(np.float32)
 result=dict(run=run,t=grid,pose=p,yaw=yaw,imu=imu100,vertical=acc,
             wheel=wheel100,ground_speed=vel,rough=rough,angular=angular,slope=slope,
             state=state,cmd=hold(ct,cmd,grid),audit=audit,calibration=calibration,
             start_epoch=base)
 deadline.log(f'official numeric {run}',start,bytes=deadline.bytes)
 return result

def ros_fields(b):
 p=0;d={}
 while p<len(b):
  n=int.from_bytes(b[p:p+4],'little');p+=4
  k,v=b[p:p+n].split(b'=',1);d[k]=v;p+=n
 return d

def ros_records(f):
 while True:
  n=f.read(4)
  if not n:return
  if len(n)<4:raise IOError('Truncated ROS record')
  head=f.read(int.from_bytes(n,'little'));h=ros_fields(head)
  m=f.read(4)
  if len(m)<4:raise IOError('Truncated ROS length')
  size=int.from_bytes(m,'little')
  data=f.read(size)
  if len(data)!=size:raise IOError('Truncated ROS payload')
  yield h,data

def decode_ros_image(data):
 import cv2
 p=0
 def num(n):
  nonlocal p
  v=int.from_bytes(data[p:p+n],'little');p+=n;return v
 _seq=num(4);secs=num(4);nanos=num(4);frame=num(4);p+=frame
 height=num(4);width=num(4);n=num(4);encoding=data[p:p+n].decode();p+=n
 big=num(1);step=num(4);length=num(4);raw=data[p:p+length]
 if height>2200 or width>4000 or len(raw)!=length or step*height!=length:
  raise RuntimeError('Invalid ROS Image dimensions/step')
 x=np.frombuffer(raw,np.uint8).reshape(height,step)
 if encoding in ('rgb8','bgr8'):
  x=x[:,:width*3].reshape(height,width,3)
  if encoding=='bgr8':x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
 elif encoding=='mono8':x=x[:,:width]
 else:raise RuntimeError('Unsupported rectified camera encoding '+encoding)
 return secs+nanos*1e-9,x

BAG_STEM={'2023-11-02-16-00-43_down_meadows':'down_meadows',
          '2023-11-14-15-02-21_figure_8':'figure_8',
          '2023-11-14-15-22-01_figure_8':'figure_8',
          '2023-11-14-15-11-49_rough_rider_turnpike_loop':'rough_rider_turnpike_loop'}

def camera_stream(run,start_epoch,seconds,root,deadline,max_frames=105):
 """Extract only the chronological first camera prefix from one official ROS bag.

 start_epoch is the numeric streams' common start, computed from their original
 timestamps. ROS Image header stamps rather than file offsets determine pairing.
 """
 import cv2,pickle
 cache=Path(root)/'selected_camera'/f'{run}_{seconds}_{max_frames}.pkl'
 if cache.exists():return pickle.loads(cache.read_bytes())
 key=f'bags/{run}/{BAG_STEM[run]}_0.bag'
 f=RemoteFile(key,Path(root)/'.range_cache'/run,deadline,block=8*1024**2)
 f.seek(0)
 if f.read(13)!=b'#ROSBAG V2.0\n':raise RuntimeError('Official first ROS bag is invalid')
 header,_=next(ros_records(f))
 if header[b'op'][0]!=3:raise RuntimeError('Missing ROS bag header')
 index_pos=int.from_bytes(header[b'index_pos'],'little')
 if not (0<index_pos<f.size):raise RuntimeError('ROS bag index offset invalid')
 # The ROS1 bag keeps its chronological chunk index at the end. Jump straight
 # to the sensor time that overlaps the KITTI numeric files; many recordings
 # have tens of seconds of camera startup before usable odometry starts.
 f.seek(index_pos);chunks=[];indexed_connections={}
 for ih,idata in ros_records(f):
  op=ih[b'op'][0]
  if op==7:indexed_connections[int.from_bytes(ih[b'conn'],'little')]=ih[b'topic'].decode()
  if op==6:
   start=int.from_bytes(ih[b'start_time'][:4],'little')+int.from_bytes(ih[b'start_time'][4:],'little')*1e-9
   end=int.from_bytes(ih[b'end_time'][:4],'little')+int.from_bytes(ih[b'end_time'][4:],'little')*1e-9
   pos=int.from_bytes(ih[b'chunk_pos'],'little')
   counts={int.from_bytes(idata[j:j+4],'little') for j in range(0,len(idata),8)}
   chunks.append((start,end,pos,counts))
 wanted={cid for cid,name in indexed_connections.items() if name in (
  '/multisense/left/image_rect_color','/multisense/right/image_rect')}
 if len(wanted)!=2:raise RuntimeError(f'Official camera topics missing: {wanted}')
 candidates=[row for row in chunks if row[1]>=start_epoch-.1 and row[0]<=start_epoch+seconds+.25
             and (row[3]&wanted)]
 if not candidates:raise RuntimeError('No indexed ROS chunk overlaps selected numeric clip')
 candidates.sort(key=lambda row:row[2])
 print(run,'indexed camera chunks:',len(candidates),'selected from',len(chunks),
       'first at',round(candidates[0][2]/1e9,3),'GB of',round(f.size/1e9,2),'GB',flush=True)
 # Downsample to 10Hz; each other 20Hz image is discarded before DINO.
 pairs={};conn=indexed_connections;frames=[];late=start_epoch+seconds+.25
 for _,_,chunk_pos,_ in candidates:
  deadline.require(100)
  f.seek(chunk_pos);outer,payload=next(ros_records(f))
  if outer[b'op'][0]!=5:raise RuntimeError('Indexed position did not point to a ROS chunk')
  if outer.get(b'compression',b'none')!=b'none':raise RuntimeError('Unexpected ROS chunk compression')
  for h,data in ros_records(io.BytesIO(payload)):
   op=h[b'op'][0]
   if op==7:
    topic=h[b'topic'].decode();conn[int.from_bytes(h[b'conn'],'little')]=topic
   if op!=2:continue
   t0=int.from_bytes(h[b'time'][:4],'little')+int.from_bytes(h[b'time'][4:],'little')*1e-9
   if t0>late:break
   topic=conn.get(int.from_bytes(h[b'conn'],'little'),None)
   if topic not in ('/multisense/left/image_rect_color','/multisense/right/image_rect'):continue
   ts,img=decode_ros_image(data)
   if not (start_epoch-.05<=ts<=start_epoch+seconds+.15):continue
   slot=round((ts-start_epoch)*10)
   if slot<0 or slot/10>seconds or abs(ts-start_epoch-slot*.1)>.046:continue
   item=pairs.setdefault(slot,{})
   if topic.endswith('image_rect_color'):item['rgb']=(ts,img)
   else:item['right']=(ts,img)
   if 'rgb' in item and 'right' in item:
    rt,left=item['rgb'];dt,right=item['right']
    if abs(rt-dt)>.025:continue
    if left.ndim!=3 or right.ndim!=2 or left.shape[:2]!=right.shape:
     raise RuntimeError('Official RGB/stereo shape mismatch')
    left=cv2.resize(left,(512,272),interpolation=cv2.INTER_AREA)
    right=cv2.resize(right,(512,272),interpolation=cv2.INTER_AREA)
    matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=64,blockSize=5,
     P1=8*3*5**2,P2=32*3*5**2,uniquenessRatio=8,speckleWindowSize=50)
    disp=matcher.compute(cv2.cvtColor(left,cv2.COLOR_RGB2GRAY),right).astype(np.float32)/16
    depth=np.divide(477.605*.5*.21,disp,out=np.full_like(disp,np.nan),where=disp>1)
    depth[(depth<.25)|(depth>30)]=np.nan
    frames.append((rt-start_epoch,left,depth))
    del pairs[slot]
    if len(frames)>=max_frames:break
   if len(pairs)>25:
    oldest=sorted(pairs)[:10]
    for k in oldest:pairs.pop(k,None)
  if len(frames)>=max_frames:break
 frames=sorted(frames,key=lambda x:x[0]);times=np.array([x[0] for x in frames])
 if len(frames)<40 or np.any(np.diff(times)<=0):raise RuntimeError(f'Insufficient synchronized RGB/stereo in {run}: {len(frames)} frames')
 cache.parent.mkdir(parents=True,exist_ok=True);result=(times,frames)
 cache.write_bytes(pickle.dumps(result,protocol=5));return result

def depth_geometry(depth):
 import cv2
 # Match facebook/dinov2-base processor: shortest side 256 then 224 center crop.
 new_w=round(depth.shape[1]*256/depth.shape[0])
 depth=cv2.resize(depth,(new_w,256),interpolation=cv2.INTER_NEAREST)
 depth=depth[16:240,(new_w-224)//2:(new_w-224)//2+224]
 good=np.isfinite(depth)
 d=np.where(good,np.clip(depth,.25,30),30).astype(np.float32)
 q=cv2.resize(d,(16,16),interpolation=cv2.INTER_AREA)
 v=cv2.resize(good.astype(np.float32),(16,16),interpolation=cv2.INTER_AREA)
 dx=np.diff(q,axis=1,append=q[:,-1:]);dy=np.diff(q,axis=0,append=q[-1:,:])
 # Depth, validity, projected height and local 3D gradients are actual stereo geometry.
 vv,uu=np.mgrid[0:16,0:16].astype(np.float32)
 scaled_fx=477.605*.5*(256/272)
 x=(uu-7.5)*q/(scaled_fx/14);y=(vv-7.5)*q/(scaled_fx/14)
 return np.stack([np.log1p(q),v,x/30,y/30,dx/30,dy/30],-1).reshape(256,6)

def camera_features(run,physics,root,deadline,dino,device,seconds,max_frames):
 import cv2,torch,pickle
 cache=Path(root)/'features'/f'{run}_{seconds}_{max_frames}_dinov2base.npz'
 if cache.exists():
  z=np.load(cache,allow_pickle=False)
  return {key:z[key] for key in z.files}
 begin=time.monotonic();start=physics['start_epoch']
 t,frames=camera_stream(run,start,seconds,root,deadline,max_frames)
 mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
 std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 tokens=[];geometry=[]
 for i in range(0,len(frames),8):
  deadline.require(100)
  batch=frames[i:i+8]
  geometry.extend([depth_geometry(f[2]) for f in batch])
  images=[]
  for f in batch:
   rgb=f[1];new_w=round(rgb.shape[1]*256/rgb.shape[0])
   resized=cv2.resize(rgb,(new_w,256),interpolation=cv2.INTER_CUBIC)
   images.append(resized[16:240,(new_w-224)//2:(new_w-224)//2+224])
  images=np.stack(images)
  x=torch.as_tensor(images,device=device).permute(0,3,1,2).float()/255
  with torch.inference_mode(),torch.autocast('cuda',enabled=device=='cuda',dtype=torch.float16):
   out=dino(pixel_values=(x-mean)/std).last_hidden_state[:,1:]
  if out.shape[1:]!=(256,768):raise RuntimeError('DINOv2-base patch count is not 256x768')
  tokens.append(out.float().cpu().numpy().astype(np.float16))
 result=dict(time=t.astype(np.float64),z=np.concatenate(tokens),geometry=np.stack(geometry).astype(np.float16))
 cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,**result)
 deadline.log('official camera+stereo+DINO '+run,begin,frames=len(frames))
 return result

# The original paper's four physical features can be computed from this ATV.
def fit_wheel_scale(training):
 ratio=[]
 for d in training:
  w=np.abs(d['wheel']).mean(1);v=d['ground_speed']
  turn=np.abs(d['state'][:,-2]);valid=(w>10)&(v>1)&(turn<.2)&(d['t']>d['calibration'])
  ratio.extend((v[valid]/w[valid]).tolist())
 if len(ratio)<30:raise RuntimeError('No reliable training-only wheel RPM/ground speed calibration')
 scale=float(np.median(ratio))
 if not (.001<scale<1):raise RuntimeError(f'Implausible wheel speed factor: {scale}')
 return scale

def attach_targets(d,wheel_scale):
 ws=np.abs(d['wheel']).mean(1)*wheel_scale;v=d['ground_speed']
 slip=(ws-v)/np.maximum(np.maximum(ws,v),.15)
 d['physical']=np.c_[d['rough'],d['angular'],slip,d['slope']].astype(np.float32)
 d['dense']=np.c_[d['vertical'],d['state'][:,-2]].astype(np.float32)
 d['wheel_scale_m_s_per_rpm']=wheel_scale
 return d

def actions(poses,yaws,times):
 delta=np.diff(poses,axis=0);c=np.cos(yaws[:-1]);s=np.sin(yaws[:-1])
 dpsi=np.diff(np.unwrap(yaws))
 return np.c_[c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1],dpsi,np.diff(times)].astype(np.float32)

def fit_normalizers(training):
 keys=['state','physical','dense']
 result={}
 for key in keys:
  allx=np.concatenate([d[key][(d['t']>=d['calibration'])&(d['t']<=d['t'][-1]-1)] for d in training])
  result[key]=(allx.mean(0).astype(np.float32),np.maximum(allx.std(0),.05).astype(np.float32))
 result['action']=(np.zeros(4,np.float32),np.array([.3,.1,.1,.1],np.float32))
 return result

def standardize(x,pair):return (x-pair[0])/pair[1]
def restore(x,pair):return x*pair[1]+pair[0]

def examples(d,features,step=.3,limit=24):
 ts=features['time'];start=max(3.0,d['calibration']+.6)
 stop=min(d['t'][-1],ts[-1])-1.05
 candidates=np.arange(start,stop,step)
 if not len(candidates):raise RuntimeError(f'No 1-second windows with 4 visual frames in {d["run"]}')
 if len(candidates)>limit:candidates=candidates[np.linspace(0,len(candidates)-1,limit).astype(int)]
 keep=[]
 for t in candidates:
  grid=t+np.arange(11)*.1;index=np.searchsorted(ts,grid,side='right')-1
  if index[0]<3 or index[-1]<0:continue
  if np.any(grid-ts[np.maximum(index,0)]>.13):continue
  keep.append((d,features,float(t),index))
 if not keep:raise RuntimeError('No synchronized camera/physical windows')
 return keep

def get_sample(item,norm,device):
 import torch
 d,features,t,ind=item;tt=t+np.arange(11)*.1
 pos=np.stack([np.interp(tt,d['t'],d['pose'][:,k]) for k in range(3)],-1)
 yaw=np.interp(tt,d['t'],d['yaw']);ac=actions(pos,yaw,tt)
 hist=hold(d['t'],d['state'],t-np.arange(49,-1,-1)*.01)
 physical=hold(d['t'],d['physical'],tt[1:])
 dense=hold(d['t'],d['dense'],t+np.arange(1,101)*.01).reshape(10,10,2)
 to=lambda x:torch.as_tensor(np.asarray(x).copy(),device=device,dtype=torch.float32)
 return dict(z=to(features['z'][ind[0]-3:ind[0]+1][None].astype(np.float32)),
   geo=to(features['geometry'][ind[0]-3:ind[0]+1][None].astype(np.float32)),
   future_z=to(features['z'][ind[1:]][None].astype(np.float32)),
   future_geo=to(features['geometry'][ind[1:]][None].astype(np.float32)),
   hist=to(standardize(hist,norm['state'])[None]),
   p=to(standardize(physical,norm['physical'])[None]),
   dense=to(standardize(dense,norm['dense'])[None]),
   action=to(standardize(ac,norm['action'])[None]),
   actual_dense=dense,actual_time=t+np.arange(1,101)*.01)

import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(22,64,7,2,3),nn.GroupNorm(8,64),nn.GELU(),
   nn.Conv1d(64,128,5,2,2),nn.GroupNorm(8,128),nn.GELU(),
   nn.Conv1d(128,128,3,2,1),nn.GroupNorm(8,128),nn.GELU(),nn.AdaptiveAvgPool1d(1))
  self.norm=nn.LayerNorm(128)
 def forward(self,h):return self.norm(self.net(h.transpose(1,2)).squeeze(-1))

class AttentionBlock(nn.Module):
 def __init__(self):
  super().__init__();self.norm1=nn.LayerNorm(768);self.norm2=nn.LayerNorm(768)
  self.qkv=nn.Linear(768,2304);self.proj=nn.Linear(768,768)
  self.mlp=nn.Sequential(nn.Linear(768,3072),nn.GELU(),nn.Dropout(.1),nn.Linear(3072,768),nn.Dropout(.1))
 def forward(self,x,mask):
  b,n,c=x.shape;q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,8,96).permute(2,0,3,1,4).unbind(0)
  a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=.1 if self.training else 0.)
  x=x+F.dropout(self.proj(a.transpose(1,2).reshape(b,n,c)),.1,self.training)
  return x+self.mlp(self.norm2(x))

class FeelWM(nn.Module):
 """Paper's 6x8x768 DINO-FILM dynamics plus measured ride-quality auxiliary head.

 Unsafe-event labels are not defined by the public KITTI package. Risk head is
 retained for an eventual labeled run, but not trained, evaluated or planned with.
 """
 def __init__(self):
  super().__init__();self.cnn=CNN();self.proj_p=nn.Sequential(nn.Linear(4,128),nn.GELU(),nn.LayerNorm(128))
  self.action=nn.Sequential(nn.Linear(4,128),nn.GELU(),nn.Linear(128,128))
  self.depth=nn.Linear(6,768);self.film=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,1536))
  nn.init.zeros_(self.film[-1].weight);nn.init.zeros_(self.film[-1].bias)
  self.time=nn.Embedding(4,768);self.layers=nn.ModuleList([AttentionBlock() for _ in range(6)])
  self.norm=nn.LayerNorm(768);self.visual=nn.Linear(768,768)
  self.physical=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Linear(256,4))
  self.ride=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Linear(256,20))
  self.risk=nn.Linear(768,1);self.risk.requires_grad_(False)
  times=torch.repeat_interleave(torch.arange(4),256)
  self.register_buffer('allowed',times[:,None]>=times[None,:],persistent=False)
 def fuse(self,z,geometry):return z+self.depth(geometry)
 def forward(self,context,p,act):
  gamma,beta=self.film(p+self.action(act)).chunk(2,-1)
  current=context[:,3]*(1+gamma[:,None])+beta[:,None]
  x=torch.cat([context[:,:3],current[:,None]],1)+self.time.weight[None,:,None]
  x=x.reshape(-1,1024,768)
  for block in self.layers:
   if self.training:
    from torch.utils.checkpoint import checkpoint
    x=checkpoint(block,x,self.allowed,use_reentrant=False)
   else:x=block(x,self.allowed)
  h=self.norm(x[:,768:]);g=h.mean(1)
  return dict(z=self.visual(h),p=self.physical(g),dense=self.ride(g).reshape(-1,10,2))

def forward_rollout(model,sample,teacher_forcing=False,train=True):
 context=model.fuse(sample['z'],sample['geo']);p=model.cnn(sample['hist']);out=[]
 for k in range(10):
  pred=model(context,p,sample['action'][:,k]);out.append(pred)
  if train and teacher_forcing:
   next_z=model.fuse(sample['future_z'][:,k],sample['future_geo'][:,k])
   p=model.proj_p(sample['p'][:,k])
  else:
   next_z=pred['z'];p=model.proj_p(pred['p'].detach())
  # Truncate gradients across rollout steps for Colab RAM, preserve recursion.
  context=torch.cat([context[:,1:],next_z.detach()[:,None]],1)
 return out

def rollout_loss(model,sample,teacher_forcing):
 output=forward_rollout(model,sample,teacher_forcing,train=True)
 losses=[]
 for k,pred in enumerate(output):
  with torch.no_grad():target=model.fuse(sample['future_z'][:,k],sample['future_geo'][:,k])
  visual=F.mse_loss(pred['z'],target)+.1*(1-F.cosine_similarity(pred['z'],target,dim=-1).mean())
  proprio=F.smooth_l1_loss(pred['p'],sample['p'][:,k])
  dense=F.smooth_l1_loss(pred['dense'],sample['dense'][:,k])
  losses.append(visual+.5*proprio+.25*dense)
 return torch.stack(losses).mean()

def run_training(model,train,val,norm,deadline,root,device,max_updates=32):
 import random,torch
 from pathlib import Path
 optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.05)
 scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda');history=[]
 model.train();random.seed(42)
 for step in range(max_updates):
  if deadline.left()<200:break
  start=time.monotonic();sample=get_sample(random.choice(train),norm,device)
  teacher=step<max_updates//4 or random.random()<(1-.5*min(1,(step/max(1,max_updates))/.75))
  optimizer.zero_grad(set_to_none=True)
  with torch.autocast('cuda',enabled=device=='cuda',dtype=torch.float16):loss=rollout_loss(model,sample,teacher)
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
  scaler.scale(loss).backward();scaler.unscale_(optimizer)
  torch.nn.utils.clip_grad_norm_(model.parameters(),1)
  scaler.step(optimizer);scaler.update()
  history.append(float(loss.detach()));print('update',step+1,'/',max_updates,'loss',round(history[-1],4),
   'seconds',round(time.monotonic()-start,1),'remaining',round(deadline.left()),flush=True)
  if step==0 and deadline.left()<200+7*(time.monotonic()-start):
   print('First update indicates fewer than 8 updates fit; resume cached data on a faster GPU.')
 if len(history)<2:raise RuntimeError('Less than two real gradient updates; no meaningful test metrics')
 target=Path(root)/'checkpoint.pt';target.parent.mkdir(parents=True,exist_ok=True)
 torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),updates=len(history),
  loss=history,norm={k:[v[0].tolist(),v[1].tolist()] for k,v in norm.items()},
  train_runs=RUNS['train'],test_run=RUNS['test'][0],architecture='FeelWM TD2 RGB+stereo 6x8x768'),target)
 print('Saved',target,'updates',len(history));return history

def evaluate(model,items,norm,device):
 model.eval();rows=[];last=None
 with torch.inference_mode():
  for item in items:
   sample=get_sample(item,norm,device);output=forward_rollout(model,sample,train=False)
   p=np.stack([o['p'][0].float().cpu().numpy() for o in output]);truth=sample['p'][0].cpu().numpy()
   dense=np.concatenate([o['dense'][0].float().cpu().numpy() for o in output])
   predicted=restore(dense,norm['dense']);real=sample['actual_dense'];ts=sample['actual_time']
   # Both signals use the identical 100Hz one-second held-out interval.
   pred_rough=float(np.std(predicted[:,0]));true_rough=float(np.std(real[:,0]))
   pred_yaw=float(np.mean(np.abs(predicted[:,1])))
   measured_yaw=hold(item[0]['t'],item[0]['yaw'],ts)
   true_yaw=ride(ts,real[:,0],measured_yaw)['yaw_vibration_rad_s']
   rows.append(dict(run=item[0]['run'],origin_s=item[2],rough_true=true_rough,rough_pred=pred_rough,
    yaw_true=true_yaw,yaw_pred=pred_yaw,physical_mae_z=float(np.mean(np.abs(p-truth)))))
   last=dict(t=ts,pred=predicted,actual=real,origin=item[2],run=item[0]['run'])
 return rows,last

# %% notebook cell 8
import json
from transformers import Dinov2Model
budget=Deadline(TOTAL_MINUTES,start=GLOBAL_START)
print('Remaining seconds after installation:',round(budget.left()))
data_start=time.monotonic()
backbone=Dinov2Model.from_pretrained('facebook/dinov2-base').to(device).eval()
backbone.requires_grad_(False)
runs={}
for split,names in RUNS.items():
    for name in names:
        budget.require(320)
        print('IMPORT:',split,name,'remaining',round(budget.left()),'seconds',flush=True)
        physics=make_physics(name,CACHE_ROOT,budget,seconds=CLIP_SECONDS)
        images=camera_features(name,physics,CACHE_ROOT,budget,backbone,device,CLIP_SECONDS,MAX_FRAMES)
        runs[name]={'split':split,'physics':physics,'images':images}
        print(name,'camera frames',len(images['time']),'last frame s',round(float(images['time'][-1]),2),flush=True)
budget.log('TOTAL relevant data import and DINO',data_start,
           downloaded_GB=round(budget.bytes/1e9,3))
del backbone;gc.collect();torch.cuda.empty_cache()
print('Available training seconds:',round(budget.left()))

# %% notebook cell 10
training=[runs[name]['physics'] for name in RUNS['train']]
scale=fit_wheel_scale(training)
for item in runs.values():attach_targets(item['physics'],scale)
norm=fit_normalizers(training)
train_items=[e for name in RUNS['train']
             for e in examples(runs[name]['physics'],runs[name]['images'],step=.20,limit=35)]
val_items=examples(runs[RUNS['validation'][0]]['physics'],
                   runs[RUNS['validation'][0]]['images'],step=.7,limit=4)
test_all=examples(runs[RUNS['test'][0]]['physics'],runs[RUNS['test'][0]]['images'],step=.2,limit=30)
chosen=min(test_all,key=lambda row:abs(row[2]-6.0))
test_items=[chosen]+[e for e in test_all[::max(1,len(test_all)//4)]
                     if abs(e[2]-chosen[2])>1e-6][:3]
print('train / validation / test origins:',len(train_items),len(val_items),len(test_items))
print('Held-out named test origin:',chosen[0]['run'],'at',round(chosen[2],2),'seconds')
print('Wheel m/s per RPM, training only:',scale)
model=FeelWM().to(device)
train_begin=time.monotonic()
loss_curve=run_training(model,train_items,val_items,norm,budget,CACHE_ROOT,device,MAX_UPDATES)
budget.log('TRAINING',train_begin,updates=len(loss_curve))

# %% notebook cell 12
import pandas as pd
eval_start=time.monotonic()
validation_rows,_=evaluate(model,val_items,norm,device)
test_rows,trace=evaluate(model,test_items,norm,device)
budget.log('VALIDATION + HELD-OUT TEST',eval_start,validation=len(validation_rows),test=len(test_rows))
print('Validation:')
display(pd.DataFrame(validation_rows).round(3))
print('Held-out test:')
display(pd.DataFrame(test_rows).round(3))

# %% notebook cell 14
import matplotlib.pyplot as plt
run_name=chosen[0]['run'];origin=chosen[2]
case=next(row for row in test_rows if row['run']==run_name and abs(row['origin_s']-origin)<1e-8)
print('One specific held-out TartanDrive 2 recording:',run_name,'at',round(origin,2),'–',round(origin+1,2),'s')
display(pd.DataFrame([{
 'source':'recorded TartanDrive 2 (this sequence)',
 'roughness (m/s²)':case['rough_true'],'yaw vibration (rad/s)':case['yaw_true']},
 {'source':'model forecast (same 100 Hz sequence)',
 'roughness (m/s²)':case['rough_pred'],'yaw vibration (rad/s)':case['yaw_pred']}]).round(3))
# Re-evaluate exactly the chosen sequence so the plotted trace matches the table row.
_,same_trace=evaluate(model,[chosen],norm,device)
fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
axes[0].plot(same_trace['t']-origin,same_trace['actual'][:,0],label='measured vertical acceleration')
axes[0].plot(same_trace['t']-origin,same_trace['pred'][:,0],label='predicted',alpha=.8)
axes[0].set_ylabel('vertical acceleration (m/s²)');axes[0].legend()
axes[1].plot(same_trace['t']-origin,same_trace['actual'][:,1],label='measured odometry heading rate')
axes[1].plot(same_trace['t']-origin,same_trace['pred'][:,1],label='predicted IMU/heading-rate proxy',alpha=.8)
axes[1].set(xlabel='seconds after test origin',ylabel='yaw rate (rad/s)');axes[1].legend()
fig.suptitle(f'TartanDrive 2 held-out recording: {run_name}, {origin:.1f}–{origin+1:.1f}s')
fig.tight_layout();fig.savefig(f'{CACHE_ROOT}/test_sequence_signals.png',dpi=140)
plt.show()

paper=[]
for (environment,route),methods in PAPER_TABLE.items():
 for name,(rough,yaw) in zip(['visual-only WM','Feel-WM without proprioception','Feel-WM'],methods):
  paper.append({'paper domain':environment,'paper route':route,'paper method':name,
                'paper roughness (m/s²)':rough,'paper yaw vibration (rad/s)':yaw})
print('Feel-WM paper Table II — contextual closed-loop route aggregates; NOT the test run above:')
display(pd.DataFrame(paper))

# %% notebook cell 16
from pathlib import Path
def plain(row):
 return {k:(float(v) if isinstance(v,(np.floating,float)) else v) for k,v in row.items()}
report={'source_dataset':'Official TartanDrive 2.0 KITTI numeric streams and ROS-bag camera prefix',
 'official_repository':'https://github.com/castacks/tartan_drive_2.0',
 'paper':'https://arxiv.org/abs/2609.19863','splits':RUNS,'seconds_per_recording':CLIP_SECONDS,
 'camera_limit_per_recording':MAX_FRAMES,'test_sequence_start_s':origin,
 'test_sequence_length_s':1.0,'train_updates':len(loss_curve),
 'GPU':torch.cuda.get_device_name(0),'wheel_calibration_train_only_m_s_per_rpm':scale,
 'stage_times':budget.phases,'elapsed_total_minutes':round((time.monotonic()-GLOBAL_START)/60,2),
 'downloaded_GB':round(budget.bytes/1e9,3),'test_rows':[plain(r) for r in test_rows],
 'validation_rows':[plain(r) for r in validation_rows],
 'sensor_audits':{name:r['physics']['audit'] for name,r in runs.items()},
 'limits':['No quadruped foot/joint/contact force in ATV data',
           'No paper-defined numerical failure threshold is provided here',
           'Recorded-action open-loop forecasting, no closed-loop simulation',
           'Initial calibration axis used for vertical acceleration']}
Path(CACHE_ROOT).mkdir(parents=True,exist_ok=True)
Path(CACHE_ROOT,'timings_and_results.json').write_text(json.dumps(report,indent=2))
print('TOTAL elapsed:',report['elapsed_total_minutes'],'minutes; real updates:',len(loss_curve),
      '; official data downloaded:',report['downloaded_GB'],'GB')
print('Saved checkpoint and results in:',CACHE_ROOT)
display(pd.DataFrame(budget.phases))
