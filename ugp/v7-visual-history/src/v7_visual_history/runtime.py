"""Opt-in hooks around the existing notebook runtime; originals stay untouched."""
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check_output_directory(foundation_root, output):
    root, dest = Path(foundation_root).resolve(), Path(output).resolve()
    if dest.is_relative_to(root) or root.is_relative_to(dest):
        raise ValueError('Output directory must not overlap the original foundation')


def record_manifest(output, manifest):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'visual_history_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Existing visual-history manifest differs; use a new output directory')
        return
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    temp.replace(path)


def normalize_rng_state(state):
    """RNG APIs require CPU byte tensors even after loading a checkpoint onto CUDA."""
    if not state:
        return state
    normalized = dict(state)
    normalized['torch_cpu'] = state['torch_cpu'].cpu()
    if 'torch_cuda' in state:
        normalized['torch_cuda'] = [value.cpu() for value in state['torch_cuda']]
    return normalized


def load_foundation_runtime(root, notebook, device):
    directory = Path(__file__).resolve().parents[1]/'v7legs_body'
    sys.path.insert(0, str(directory))
    from foundation import load_runtime
    return load_runtime(root, notebook, device)


def install_visual_history(rt, foundation_root, frames=6):
    """Patch only this newly loaded definition dictionary, including V7-F extraction."""
    import torch
    from .data import build_history
    from .model import V7VisualHistory
    if rt.get('VISUAL_HISTORY_INSTALLED'):
        raise ValueError('Visual history is already installed in this runtime')
    original_factory = rt['make_model_for_record']
    original_sample = rt['make_v7_sample']
    original_forward = rt['model_forward_v7']
    original_force = rt['v7f_extract_base_context']
    original_restore_rng = rt['_restore_rng_state']
    checkpoint_root = Path(foundation_root)/'storage/results'

    def make_model(rec, steps, horizons):
        short = len(horizons) == 1 and abs(float(horizons[0])-.1) < 1e-7
        expected = [.1] if short else [i/10 for i in range(2, 11)]
        if len(horizons) != len(expected) or any(abs(float(a)-b) > 1e-7 for a, b in zip(horizons, expected)):
            raise ValueError('Use the existing short or long V7 horizons')
        base = original_factory(rec, steps, horizons)
        path = checkpoint_root/('V7_PRIOR_SHORT.pt' if short else 'V7_PRIOR_LONG.pt')
        base.load_state_dict(torch.load(path, map_location=rt['DEVICE'], weights_only=False))
        return V7VisualHistory(base, frames).to(rt['DEVICE'])

    def make_sample(data, visual, stats, physics, idx, steps, horizons, norm=None):
        sample = original_sample(data, visual, stats, physics, idx, steps, horizons, norm)
        sample['visual_history'] = build_history(data, visual, idx, steps, rt['DEVICE'], frames)
        return sample

    def forward(model, sample):
        return model.forward_sample(sample) if isinstance(model, V7VisualHistory) else original_forward(model, sample)

    def restore_rng(state):
        return original_restore_rng(normalize_rng_state(state))

    @torch.inference_mode()
    def force_context(model, sample, stats):
        if not isinstance(model, V7VisualHistory):
            return original_force(model, sample, stats)
        model.eval()
        x = model.features(sample)
        pstd = torch.as_tensor(stats['P_std'], device=x.device, dtype=x.dtype)
        baseline = torch.as_tensor(sample['baseline_p_raw'], device=x.device, dtype=x.dtype)[None]
        residual = model.base.tau_force_head(x)[..., 3:6]
        return dict(x=x.detach(), baseF=(baseline[..., 9:12]+residual*pstd[9:12]).detach(),
                    pcontact=x[..., -model.n_regimes-1].detach(), regime=x[..., -model.n_regimes:].detach())

    rt.update(make_model_for_record=make_model, make_v7_sample=make_sample, model_forward_v7=forward,
              v7f_extract_base_context=force_context, _restore_rng_state=restore_rng,
              VISUAL_HISTORY_INSTALLED=True)
    return dict(original_factory=original_factory, original_sample=original_sample, original_forward=original_forward)
