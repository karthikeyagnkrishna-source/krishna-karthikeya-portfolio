"""Read checkpoint shape metadata without importing Torch or executing saved globals."""
from pathlib import Path
import collections
import io
import json
import math
import pickle
import zipfile

class TensorShape:
    def __init__(self,shape):self.shape=tuple(shape)
class Opaque:
    def __init__(self,*args,**kwargs):self.args=args
    def __setstate__(self,state):self.state=state
def rebuild_tensor(storage,offset,size,stride,*args,**kwargs):return TensorShape(size)
def rebuild_parameter(data,*args,**kwargs):return data
class ShapeReader(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='collections' and name=='OrderedDict':return collections.OrderedDict
        if module=='torch._utils' and name in {'_rebuild_tensor','_rebuild_tensor_v2','_rebuild_tensor_v3'}:return rebuild_tensor
        if module=='torch._utils' and name.startswith('_rebuild_parameter'):return rebuild_parameter
        # All other saved global names become inert placeholders, never imported.
        return Opaque
    def persistent_load(self,pid):return Opaque(pid)

base=Path('/path/to/ugp/V7F_4080_Run/storage/V8_contactmap')
out={}
for p in base.glob('*/*.pt'):
    with zipfile.ZipFile(p) as z:
        name=next(n for n in z.namelist() if n.endswith('/data.pkl'))
        raw=z.read(name)
    ck=ShapeReader(io.BytesIO(raw)).load()
    sd=ck.get('model',ck.get('state_dict',ck))
    shapes={k:list(v.shape) for k,v in sd.items() if isinstance(v,TensorShape)}
    out[str(p)]={'num_tensors':len(shapes),'state_numel':sum(math.prod(s) for s in shapes.values()),'tensor_shapes':shapes,'method':'inert static pickle metadata parser; no saved globals imported or executed'}
print(json.dumps(out))
