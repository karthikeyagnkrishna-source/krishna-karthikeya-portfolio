"""Prepare only excluded-recording input commands; no model fit or scaler changes."""
import argparse
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2]/'v7legs_body'),str(Path(__file__).resolve().parents[2]/'v7feedback')]
import numpy as np
from foundation import load_runtime,sha256
from command_data import verified_command_cache,extract_commands
from recording_source import recording
from data import atomic_json
from train_joint_specialists import Reporter

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    notebook=Path(__file__).resolve().parents[2]/'v7legs_body/foundation.ipynb'
    rt=load_runtime(args.root,notebook,'cpu');report=Reporter(args.output)
    manifest=json.loads((args.output/'manifest.json').read_text());commands=args.output/'commands'
    mapping=np.asarray(json.loads((commands/'manifest.json').read_text())['mapping'])
    evidence={}
    for n,(env,run) in enumerate(manifest['heldout_trajectories'],1):
        report('evaluation_input_preparation',recording=f'{env}/{run}',recordings_completed=n-1,recordings_total=4,training_trajectories=28)
        source=rt['CACHE_ROOT']/f'{env}_{run}_physics.npz'
        with np.load(source) as z:physics={k:z[k] for k in z.files}
        path=commands/f'{env}_{run}_commands.npz'
        if not verified_command_cache(path,physics,mapping):
            with recording(rt['ROOT'],env,run,args.output/'temporary_recordings') as bag:extract_commands(bag,physics,mapping,path)
        evidence[f'{env}/{run}']=dict(cache_sha256=sha256(path),metadata_sha256=sha256(path.with_suffix('.json')))
    atomic_json(commands/'evaluation_sources.json',dict(recordings=evidence,note='Input preparation only, excluded from all optimization and scales. Fixed training hardware mapping.'))
    report('evaluation_inputs_ready',training_trajectories=28)

if __name__=='__main__':main()
