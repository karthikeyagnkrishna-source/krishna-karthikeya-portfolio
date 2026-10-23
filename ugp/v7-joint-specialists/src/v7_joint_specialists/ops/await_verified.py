"""Continue this authorized pipeline once both real branches pass verification."""
import argparse
import json
import os
from pathlib import Path
import time

def verified_status(status,pid):
    if status.get('pid')!=pid:raise ValueError('Status belongs to a different preflight process')
    if status.get('phase')=='failed':raise RuntimeError('Real verification failed: '+status.get('error','unknown error'))
    return status.get('phase')=='verification_complete'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    print('Waiting for matching real short/long verification',flush=True)
    path=args.run/'results/status.json'
    while True:
        if path.exists() and verified_status(json.loads(path.read_text()),args.pid):break
        if not Path('/proc'/Path(str(args.pid))).exists():raise RuntimeError('Verification exited without successful completion')
        time.sleep(15)
    print('Both branches verified; preparing excluded input commands then choosing a free GPU',flush=True)
    launcher=args.run/'code/v7_joint_specialists/ops/launch_verified.sh'
    os.execv('/bin/bash',['bash',str(launcher)])

if __name__=='__main__':main()
