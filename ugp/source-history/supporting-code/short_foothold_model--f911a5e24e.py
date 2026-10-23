# V7 SHORT Spatial-Temporal Foothold Memory — self-contained notebook cell.
# Requires the unchanged baseline cells 1–10 and the preceding NumPy utility cell.

SHORT_FM_HORIZON_S = 0.6
SHORT_FM_MAX_FRAME_AGE_S = 0.12
SHORT_FM_MAX_FRAME_GAP_S = 0.20
SHORT_FM_GRID = 8
SHORT_FM_NO_TOUCH_CELL = 64
SHORT_FM_CAMERA_DELAY_S = 0.20  # train-derived upper bound (0.175 s) plus 25 ms margin
SHORT_FM_CLOCK_WARMUP_S = 21.0


def short_fm_validate_and_shift_clock(rec):
    """Calibrate from a causal 3-20 s prefix; delay images conservatively."""
    if rec.get('_short_fm_clock_applied'):
        return rec['_short_fm_clock_audit']
    t = np.asarray(rec['data']['time'], np.float64)
    b = np.asarray(rec['data']['B'][:, 6:9], np.float64)
    ct = np.asarray(rec['manifest']['camera_times'], np.float64)
    pose = np.asarray(rec['manifest']['poses'], np.float64)
    D = np.diag([1., -1., -1.])
    v_world = np.gradient(pose[:, :3] @ D, ct, axis=0)
    rot = D @ Rotation.from_quat(pose[:, 3:]).as_matrix() @ D
    v_base = np.einsum('nji,nj->ni', rot, v_world)
    prefix = (ct >= 3.) & (ct <= 20.)
    if prefix.sum() < 100 or t[-1] < SHORT_FM_CLOCK_WARMUP_S + SHORT_FM_HORIZON_S:
        raise RuntimeError('Insufficient causal clock-calibration prefix')
    scores = []
    for lag in np.arange(0., .301, .005):
        q = ct[prefix] + lag
        reference = np.column_stack([np.interp(q, t, b[:, k]) for k in range(3)])
        error = float(np.median(np.linalg.norm(v_base[prefix] - reference, axis=1)))
        scores.append((error, float(lag)))
    error, lag = min(scores)
    improvement = 1. - error / max(scores[0][0], 1e-8)
    if lag > .18 or error > .08 or improvement < .4:
        raise RuntimeError(f'Camera/robot clock validation failed for {rec["environment"]}/{rec["run"]}: lag={lag:.3f}s, error={error:.3f}m/s, improvement={improvement:.2f}')
    if not np.allclose(rec['visual']['camera_times'], ct, atol=1e-6):
        raise RuntimeError('Visual cache and manifest camera clocks differ')
    rec['manifest'] = dict(rec['manifest'])
    rec['visual'] = dict(rec['visual'])
    rec['manifest']['camera_times'] = ct + SHORT_FM_CAMERA_DELAY_S
    rec['visual']['camera_times'] = ct + SHORT_FM_CAMERA_DELAY_S
    audit = dict(environment=rec['environment'], run=rec['run'], prefix_end_s=20., estimated_lag_s=lag,
                 applied_delay_s=SHORT_FM_CAMERA_DELAY_S, median_velocity_error_m_s=error,
                 improvement_vs_zero=improvement)
    rec['_short_fm_clock_audit'] = audit
    rec['_short_fm_clock_applied'] = True
    return audit


def short_fm_robot_rows(data, start, stop):
    return np.concatenate([
        data['P'][start:stop].reshape(stop-start, 48), data['B'][start:stop],
        data['U'][start:stop], data['L_LEG'][start:stop].reshape(stop-start, 12),
        data['E'][start:stop].reshape(stop-start, 12),
    ], axis=1).astype(np.float32)


def short_fm_fit_robot_stats(records):
    sums = np.zeros(90, np.float64); squares = np.zeros(90, np.float64); count = 0
    for rec in records:
        t = rec['data']['time']; cutoff = min(rec['ttests'])
        n = int(np.searchsorted(t, cutoff, side='right'))
        x = short_fm_robot_rows(rec['data'], 0, n).astype(np.float64)
        sums += x.sum(0); squares += np.square(x).sum(0); count += len(x)
    if count < 100:
        raise RuntimeError('Insufficient training-only robot statistics')
    mean = sums/count; std = np.sqrt(np.maximum(squares/count-mean*mean, 1e-8))
    return mean.astype(np.float32), np.maximum(std, .01).astype(np.float32)


def short_fm_frame_grid(rec, frame_j):
    cache = rec.setdefault('_short_fm_grid_cache', {})
    if frame_j in cache:
        return cache[frame_j]
    man = rec['manifest']
    x0, y0, x1, y1 = map(int, man['crop_boxes'][frame_j])
    pose = man['poses'][frame_j]
    Rlw, Rwc = heading_frame_from_pose(pose)
    rep = np.asarray(rec['visual']['GEOM'][frame_j], np.float32)
    # The cached Utonia/PTv3 tokens contain the original RGB-D ground-point
    # coordinates in the final ten channels. This remains usable after the
    # baseline safely removes auxiliary runs' redownloadable raw frames.
    local = rep[:, -10:-7]
    world_delta = local @ Rlw
    camera = world_delta @ Rwc
    depth = camera[:, 0]
    safe = np.isfinite(rep).all(1) & (depth > 1e-6)
    u = 320. + 320. * camera[:, 1] / np.maximum(depth, 1e-6)
    v = 320. + 320. * camera[:, 2] / np.maximum(depth, 1e-6)
    inside = safe & (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
    cell_x = np.floor(8. * (u-x0) / max(x1-x0, 1)).astype(np.int64)
    cell_y = np.floor(8. * (v-y0) / max(y1-y0, 1)).astype(np.int64)
    world = world_delta + np.asarray(pose[:3], np.float32)
    xyz = np.zeros((64, 3), np.float32); valid = np.zeros(64, np.float32)
    for c in range(64):
        chosen = inside & (cell_y*8+cell_x == c)
        if chosen.any():
            xyz[c] = np.median(world[chosen], axis=0)
            valid[c] = 1.
    if not valid.any():
        raise ValueError('RGB-D ground grid has no valid cell')
    cache[frame_j] = (xyz, valid)
    return xyz, valid


def short_fm_sample(rec, idx, calibration, robot_stats, events, horizon_s=SHORT_FM_HORIZON_S):
    data, vis = rec['data'], rec['visual']
    anchor = float(data['time'][idx]); dt = float(np.median(np.diff(data['time'])))
    H = int(round(SHORT_CONTEXT_SEC/dt))
    if idx-H+1 < 0:
        raise ValueError('Insufficient causal robot history')
    ids = visual_window_indices(vis['camera_times'], anchor, SHORT_FM_MAX_FRAME_AGE_S, SHORT_FM_MAX_FRAME_GAP_S)
    if not np.array_equal(ids, np.arange(ids[-1]-4, ids[-1]+1)):
        raise RuntimeError('Visual window is not five consecutive frames')
    flow = np.asarray(vis['FLOW'][ids[-1]], np.float32)
    if flow.shape[:2] != (4, 64):
        raise RuntimeError('Expected exactly four cached optical-flow pairs ending at current frame')
    dino = np.asarray(vis['DINO'][ids], np.float32)
    conv = np.asarray(vis['CONV'][ids], np.float32)
    geom = np.asarray(vis['GEOM'][ids], np.float32)
    grid_xyz = []; grid_valid = []
    camera_xyz = np.asarray(vis['poses'][ids[-1], :3], np.float32)
    for j in ids:
        xyz, valid = short_fm_frame_grid(rec, int(j))
        grid_xyz.append(xyz-camera_xyz); grid_valid.append(valid)
    grid_xyz = np.asarray(grid_xyz, np.float32)
    grid_valid = np.asarray(grid_valid, np.float32)
    # Reindex older image patches into current WORLD-metric cells. Image pixel
    # indices alone do not track terrain as the camera moves.
    current_world = grid_xyz[-1] + camera_xyz
    # Utonia cache stores representative point XYZ as the first 3 of the last
    # 10 raw channels (XYZ, normal, color, roughness). Align its 64 point tokens
    # to metric cells rather than treating point order as image patch order.
    aligned_geom = np.zeros_like(geom)
    for f, frame_j in enumerate(ids):
        Rlw, _ = heading_frame_from_pose(vis['poses'][frame_j])
        rep_world = geom[f, :, -10:-7] @ Rlw + np.asarray(vis['poses'][frame_j, :3])
        counts = np.zeros(64, np.int32)
        for point, feature in zip(rep_world, geom[f]):
            dest = cell_for_point(point, current_world, grid_valid[-1].astype(bool))
            if dest is not None:
                aligned_geom[f, dest] += feature; counts[dest] += 1
        keep = counts > 0
        aligned_geom[f, keep] /= counts[keep, None]
    geom = aligned_geom
    for f in range(4):
        old_dino = dino[f].copy(); old_conv = conv[f].copy()
        old_flow = flow[f-1].copy() if f > 0 else None
        old_xyz = grid_xyz[f].copy(); old_valid = grid_valid[f].copy()
        dino[f] = 0.; conv[f] = 0.; grid_valid[f] = 0.
        if f > 0:
            flow[f-1] = 0.
        for cell in range(64):
            if grid_valid[-1, cell] < .5:
                continue
            source = cell_for_point(current_world[cell], old_xyz+camera_xyz, old_valid.astype(bool))
            if source is not None:
                dino[f, cell] = old_dino[source]; conv[f, cell] = old_conv[source]
                if f > 0:
                    flow[f-1, cell] = old_flow[source]
                grid_xyz[f, cell] = old_xyz[source]; grid_valid[f, cell] = 1.
    ages = (anchor-np.asarray(vis['camera_times'][ids], np.float32))[:, None, None]
    time_grid = np.broadcast_to(ages, (5, 64, 1))
    grid = np.concatenate([grid_xyz, time_grid, grid_valid[..., None]], axis=-1)
    touches, experience = memory_at(anchor, events, current_world, grid_valid[-1].astype(bool), camera_xyz)
    foot_world = np.stack([foot_world_at(rec, idx, leg, calibration, ANYMAL_D_FK) for leg in range(4)])
    mean, std = robot_stats
    robot = (short_fm_robot_rows(data, idx-H+1, idx+1)-mean)/std
    target = target_at(anchor, horizon_s, events, current_world, grid_valid[-1].astype(bool), camera_xyz)
    return dict(
        robot=robot, dino=dino, conv=conv, geom=geom, flow=flow,
        grid=grid, touches=touches, experience=experience,
        foot_xyz=(foot_world-camera_xyz).astype(np.float32), target=target,
        anchor_t=anchor, frame_ids=ids, camera_xyz=camera_xyz,
    )


class V7ShortFootholdMemory(nn.Module):
    """Separate V7 SHORT extension; the original V7/V7-F weights and heads stay intact."""
    def __init__(self, history_steps, dino_dim, conv_dim, geom_dim):
        super().__init__()
        self.history_steps = int(history_steps)
        self.robot_encoder = CausalHistoryEncoder(90, history_steps, 128, 2, 4)
        self.dino_proj = nn.Sequential(nn.LayerNorm(dino_dim), nn.Linear(dino_dim, 96), nn.GELU())
        self.conv_proj = nn.Sequential(nn.LayerNorm(conv_dim), nn.Linear(conv_dim, 96), nn.GELU())
        self.geom_proj = nn.Sequential(nn.LayerNorm(geom_dim), nn.Linear(geom_dim, 64), nn.GELU())
        self.geom_temporal = nn.GRU(64, 64, batch_first=True)
        self.temporal_grid = nn.GRU(197, 192, batch_first=True)  # DINO96+Conv96+XYZ3+age1+valid1
        self.flow_proj = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, 32), nn.GELU())
        self.flow_temporal = nn.GRU(32, 32, batch_first=True)
        self.experience_proj = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 32), nn.GELU())
        self.cell_proj = nn.Sequential(nn.LayerNorm(192+64+32+32+4), nn.Linear(324, 256), nn.GELU())
        self.touch_proj = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, 64), nn.GELU())
        self.leg_id = nn.Embedding(4, 32)
        self.query_proj = nn.Sequential(nn.LayerNorm(128+64+32+3), nn.Linear(227, 256), nn.GELU())
        self.cross = nn.MultiheadAttention(256, 8, batch_first=True)
        self.cell_score = nn.Linear(256, 256)
        self.no_touch = nn.Linear(256, 1)
        self.position = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.touch_time = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, robot, dino, conv, geom, flow, grid, touches, experience, foot_xyz):
        B = robot.shape[0]
        if dino.shape[1:3] != (5, 64) or conv.shape[1:3] != (5, 64):
            raise ValueError('Expected five synchronized 8x8 visual grids')
        if geom.shape[1] != 5 or flow.shape[1:3] != (4, 64):
            raise ValueError('Expected five PTv3 frames and four flow pairs')
        valid = grid[:, -1, :, 4] > .5
        if not bool(valid.any(dim=1).all()):
            raise ValueError('Every sample needs a valid current ground cell')
        x = torch.cat([self.dino_proj(dino), self.conv_proj(conv), grid], -1)
        x = x.permute(0, 2, 1, 3).reshape(B*64, 5, 197)
        temporal = self.temporal_grid(x)[1][-1].reshape(B, 64, 192)
        geom_sequence = self.geom_proj(geom) * (geom.abs().sum(-1, keepdim=True) > 0).to(geom.dtype)
        geom_sequence = geom_sequence.permute(0, 2, 1, 3).reshape(B*64, 5, 64)
        geom_frame = self.geom_temporal(geom_sequence)[1][-1].reshape(B, 64, 64)
        flow_sequence = self.flow_proj(flow).permute(0, 2, 1, 3).reshape(B*64, 4, 32)
        flow_cell = self.flow_temporal(flow_sequence)[1][-1].reshape(B, 64, 32)
        exp_cell = self.experience_proj(experience)
        cells = self.cell_proj(torch.cat([temporal, geom_frame, flow_cell, exp_cell, grid[:, -1, :, :4]], -1))
        robot_state = self.robot_encoder(robot)[:, None].expand(-1, 4, -1)
        touch = self.touch_proj(touches).masked_fill(touches[..., 5:6] < .5, 0).sum(2) / touches[..., 5].sum(2).clamp_min(1)[..., None]
        legid = self.leg_id(torch.arange(4, device=robot.device))[None].expand(B, -1, -1)
        query = self.query_proj(torch.cat([robot_state, touch, legid, foot_xyz], -1))
        # Location bias uses measured foot and metric 3D cell positions; invalid cells are masked.
        distance = torch.cdist(foot_xyz, grid[:, -1, :, :3])
        bias = (-distance / .5).masked_fill(~valid[:, None, :], -1e4)
        mask = bias[:, None].expand(-1, 8, -1, -1).reshape(B*8, 4, 64)
        attended, _ = self.cross(query, cells, cells, attn_mask=mask, need_weights=False)
        fused = query + attended
        logits = torch.einsum('bld,bsd->bls', self.cell_score(fused), cells) / 16.
        logits = logits.masked_fill(~valid[:, None, :], -1e4)
        logits = torch.cat([logits, self.no_touch(fused)], -1)
        location = torch.softmax(logits[..., :64], -1) @ grid[:, -1, :, :3]
        xyz = location + self.position(fused)
        tau = SHORT_FM_HORIZON_S * torch.sigmoid(self.touch_time(fused).squeeze(-1))
        return dict(cell_logits=logits, xyz=xyz, tau=tau)


def short_fm_loss(output, target):
    valid = target['valid'] > .5
    positive = (target['positive'] > .5) & valid
    if not bool(valid.any()):
        raise ValueError('No valid foothold labels in batch')
    cls = F.cross_entropy(output['cell_logits'][valid], target['cell'][valid])
    if bool(positive.any()):
        xyz = F.smooth_l1_loss(output['xyz'][positive], target['xyz'][positive], beta=.1)
        timing = F.smooth_l1_loss(output['tau'][positive], target['tau'][positive], beta=.05)
    else:
        xyz = output['xyz'].sum()*0.; timing = output['tau'].sum()*0.
    return cls + 2.*xyz + timing


def short_fm_to_torch(sample, device):
    x = {}
    for k in ('robot', 'dino', 'conv', 'geom', 'flow', 'grid', 'touches', 'experience', 'foot_xyz'):
        x[k] = torch.as_tensor(sample[k][None], dtype=torch.float32, device=device)
    target = {}
    for k, v in sample['target'].items():
        dtype = torch.long if k == 'cell' else torch.float32
        target[k] = torch.as_tensor(v[None], dtype=dtype, device=device)
    return x, target


def short_fm_training_anchors(rec, horizon_s, hop_s=.3):
    data, vis = rec['data'], rec['visual']
    cutoff = float(min(rec['ttests']))
    first = max(float(data['time'][0])+SHORT_CONTEXT_SEC, float(vis['camera_times'][4]), SHORT_FM_CLOCK_WARMUP_S)
    anchors = []
    for desired in np.arange(first, cutoff-horizon_s-.05+1e-9, hop_s):
        idx = int(np.searchsorted(data['time'], desired, side='right')-1)
        if idx < 0 or not target_end_ok(float(data['time'][idx]), horizon_s, cutoff):
            continue
        try:
            visual_window_indices(vis['camera_times'], float(data['time'][idx]), SHORT_FM_MAX_FRAME_AGE_S, SHORT_FM_MAX_FRAME_GAP_S)
        except ValueError:
            continue
        anchors.append(idx)
    return sorted(set(anchors))


def short_fm_train(train_records, calibration, epochs=3, lr=2e-4):
    if len(train_records) != 28 or {(r['environment'], r['run']) for r in train_records} & set(STRICT_HOLDOUTS.values()):
        raise RuntimeError('Foothold trainer requires exactly 28 training runs and zero strict-holdout overlap')
    stats = short_fm_fit_robot_stats(train_records)
    first = train_records[0]; dt = float(np.median(np.diff(first['data']['time'])))
    H = int(round(SHORT_CONTEXT_SEC/dt))
    vis = first['visual']
    model = V7ShortFootholdMemory(H, vis['DINO'].shape[-1], vis['CONV'].shape[-1], vis['GEOM'].shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    audit = []
    for epoch in range(epochs):
        model.train()
        for rec in train_records:
            rows = event_records(rec, calibration, ANYMAL_D_FK)
            count = 0; skipped = 0; positives = 0; losses = []
            cutoff = float(min(rec['ttests']))
            for idx in short_fm_training_anchors(rec, SHORT_FM_HORIZON_S):
                anchor = float(rec['data']['time'][idx])
                if not target_end_ok(anchor, SHORT_FM_HORIZON_S, cutoff):
                    raise RuntimeError('Foothold training label crosses training cutoff')
                try:
                    sample = short_fm_sample(rec, idx, calibration, stats, rows)
                except ValueError:
                    skipped += 1; continue
                if int(round(SHORT_CONTEXT_SEC / float(np.median(np.diff(rec['data']['time']))))) != H:
                    raise RuntimeError('Robot history length differs across runs')
                x, target = short_fm_to_torch(sample, DEVICE)
                if not bool((target['valid'] > .5).any()):
                    skipped += 1
                    continue
                positives += int(((target['positive'] > .5) & (target['valid'] > .5)).sum().item())
                opt.zero_grad(set_to_none=True)
                out = model(**x); loss = short_fm_loss(out, target)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Non-finite foothold loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
                count += 1; losses.append(float(loss.detach().cpu()))
            audit.append(dict(epoch=epoch+1, environment=rec['environment'], run=rec['run'], cutoff_s=cutoff,
                              trained=count, skipped=skipped, visible_touchdowns=positives,
                              loss=float(np.mean(losses)) if losses else np.nan))
            if not count:
                raise RuntimeError(f'No valid foothold training samples: {rec["environment"]}/{rec["run"]}')
            print(f'SHORT FM epoch {epoch+1} {rec["environment"]}/{rec["run"]}: {count} train, {skipped} skipped')
    audit = pd.DataFrame(audit)
    if int(audit.visible_touchdowns.sum()) == 0:
        raise RuntimeError('No verified visible touchdown labels in the 28-run training set')
    return model, stats, audit


def short_fm_evaluate(model, stats, holdout_records, calibration, hop_s=.5):
    train_keys = set(x for runs in EXPANDED_TRAIN_RUNS.values() for x in runs)
    expected = set(STRICT_HOLDOUTS.values())
    actual = {(r['environment'], r['run']) for r in holdout_records}
    if actual != expected or actual & train_keys:
        raise RuntimeError('Evaluation requires exactly the four strict unseen trajectories')
    model.eval(); output_rows = []; skipped = []
    for rec in holdout_records:
        events = event_records(rec, calibration, ANYMAL_D_FK)
        t = rec['data']['time']; start = max(float(t[0])+SHORT_CONTEXT_SEC, float(rec['visual']['camera_times'][4]), SHORT_FM_CLOCK_WARMUP_S)
        for desired in np.arange(start, float(t[-1])-SHORT_FM_HORIZON_S-.05, hop_s):
            idx = int(np.searchsorted(t, desired, side='right')-1)
            if idx < 0:
                continue
            try:
                sample = short_fm_sample(rec, idx, calibration, stats, events)
            except ValueError as exc:
                skipped.append(dict(environment=rec['environment'], run=rec['run'], anchor_s=desired, reason=str(exc)))
                continue
            x, _ = short_fm_to_torch(sample, DEVICE)
            with torch.no_grad():
                out = model(**x)
                prob = out['cell_logits'].softmax(-1)[0].cpu().numpy()
                pos = out['xyz'][0].cpu().numpy()
                tau = out['tau'][0].cpu().numpy()
            y = sample['target']
            for leg in range(4):
                if y['valid'][leg] < .5:
                    skipped.append(dict(environment=rec['environment'], run=rec['run'], anchor_s=sample['anchor_t'], leg=leg, reason='touchdown outside valid visible grid'))
                    continue
                pred = int(np.argmax(prob[leg]))
                event = bool(y['positive'][leg])
                output_rows.append(dict(environment=rec['environment'], run=rec['run'], anchor_s=sample['anchor_t'], leg=leg,
                    actual_cell=int(y['cell'][leg]), predicted_cell=pred, predicted_no_touch_prob=float(prob[leg, 64]),
                    event=event, cell_correct=bool(pred == int(y['cell'][leg])),
                    position_error_m=float(np.linalg.norm(pos[leg]-y['xyz'][leg])) if event else np.nan,
                    timing_error_s=float(abs(tau[leg]-y['tau'][leg])) if event else np.nan))
    if not output_rows:
        raise RuntimeError('No valid strict-holdout evaluation samples')
    return pd.DataFrame(output_rows), pd.DataFrame(skipped)


def short_fm_load_holdouts(calibration):
    """Prepare the exact four holdouts for read-only evaluation after model fitting."""
    records = []
    for terrain, (env, run) in STRICT_HOLDOUTS.items():
        base = ROOT/env/'Data_anymal'/run
        try:
            data = extract_run_physics(base, run)
            manifest = build_frame_manifest(base, env, run, data)
            visual = load_visual_bundle(env, run, manifest)
        except (FileNotFoundError, RuntimeError):
            base = download_run_v7(env, run)
            data = extract_run_physics(base, run)
            manifest = build_frame_manifest(base, env, run, data)
            build_dinov3_cache(base, env, run, manifest)
            build_convnext_cache(base, env, run, manifest)
            build_utonia_cache(base, env, run, manifest)
            build_waft_cache(base, env, run, manifest)
            visual = load_visual_bundle(env, run, manifest)
        rec = dict(terrain_type=terrain, environment=env, run=run, base=base, data=data,
                   manifest=manifest, visual=visual)
        short_fm_validate_and_shift_clock(rec)
        # Validate the dataset pose transform on an actual run sample before evaluation.
        i = int(np.searchsorted(data['time'], visual['camera_times'][4], side='left'))
        foot_world_at(rec, i, 0, calibration, ANYMAL_D_FK)
        records.append(rec)
    return records


def short_fm_run(calibration_json):
    """Opt-in full run; missing calibration or missing prepared baseline records fails closed."""
    calibration = load_verified_calibration(calibration_json)
    if 'TRAIN_RECORDS' not in globals() or len(TRAIN_RECORDS) != 28:
        raise RuntimeError('Run V7 baseline data preparation before the foothold extension')
    clock_audit = [short_fm_validate_and_shift_clock(rec) for rec in TRAIN_RECORDS]
    result_root = OUT_ROOT/'V7_SHORT_FOOTHOLD_MEMORY'; result_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clock_audit).to_csv(result_root/'TRAIN_CLOCK_AUDIT.csv', index=False)
    model, stats, audit = short_fm_train(TRAIN_RECORDS, calibration)
    torch.save(dict(state_dict=model.state_dict(), robot_mean=stats[0], robot_std=stats[1],
                    horizon_s=SHORT_FM_HORIZON_S, calibration_source=calibration['verified_source'],
                    calibration_sha256=hashlib.sha256(Path(calibration_json).read_bytes()).hexdigest(),
                    train_keys=sorted((r['environment'], r['run']) for r in TRAIN_RECORDS)),
               result_root/'V7_SHORT_FM.pt')
    audit.to_csv(result_root/'TRAIN_AUDIT.csv', index=False)
    # The four strict holdouts are loaded only after training returns.
    holdouts = short_fm_load_holdouts(calibration)
    pd.DataFrame([r['_short_fm_clock_audit'] for r in holdouts]).to_csv(result_root/'HOLDOUT_CLOCK_AUDIT.csv', index=False)
    raw, skipped = short_fm_evaluate(model, stats, holdouts, calibration)
    raw.to_csv(result_root/'STRICT_HOLDOUT_RAW.csv', index=False)
    skipped.to_csv(result_root/'STRICT_HOLDOUT_SKIPPED.csv', index=False)
    summary = raw.groupby(['environment', 'run'], as_index=False).agg(
        n=('event', 'size'), touchdown_count=('event', 'sum'), cell_accuracy=('cell_correct', 'mean'),
        position_MAE_m=('position_error_m', 'mean'), timing_MAE_s=('timing_error_s', 'mean'))
    summary.to_csv(result_root/'STRICT_HOLDOUT_SUMMARY.csv', index=False)
    display(summary)
    return model, summary, audit, skipped
