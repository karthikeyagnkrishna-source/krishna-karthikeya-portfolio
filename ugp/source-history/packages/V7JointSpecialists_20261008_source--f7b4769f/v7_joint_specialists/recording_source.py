"""Recover only missing public ROS bags, one isolated temporary recording at a time."""
from contextlib import contextmanager
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

def fetch_archive(env,run,path):
    # Same public repository and rosbags key used by installed TartanGround downloader.
    import requests
    key=f'{env}/Data_anymal/{run}/rosbags.zip'
    url='https://huggingface.co/datasets/theairlabcmu/TartanGround/resolve/main/'+key
    with requests.get(url,stream=True,timeout=(30,120)) as response:
        response.raise_for_status()
        with Path(path).open('wb') as f:
            for chunk in response.iter_content(1024*1024):
                if chunk:f.write(chunk)

@contextmanager
def recording(root,env,run,workspace,fetch=None):
    if not re.fullmatch(r'[A-Za-z0-9_]+',env) or not re.fullmatch(r'P[0-9]+',run):raise ValueError('Invalid recording identity')
    original=Path(root)/env/'Data_anymal'/run/'rosbags'/f'{run}.bag'
    if original.is_file():
        yield original;return
    workspace=Path(workspace);workspace.mkdir(parents=True,exist_ok=True)
    # TemporaryDirectory owns only newly created files; existing recordings are never removed.
    with tempfile.TemporaryDirectory(prefix=f'{env}_{run}_',dir=workspace) as directory:
        temp=Path(directory);archive=temp/'rosbags.zip'
        (fetch or fetch_archive)(env,run,archive)
        with zipfile.ZipFile(archive) as z:
            bags=[i for i in z.infolist() if not i.is_dir() and Path(i.filename).name==f'{run}.bag']
            if len(bags)!=1:raise ValueError('Archive does not contain exactly the requested ROS recording')
            bag=temp/f'{run}.bag'
            with z.open(bags[0]) as src,bag.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
        yield bag
