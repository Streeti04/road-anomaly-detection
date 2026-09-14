"""Shared AE training; Mini default, explicit Stage-2 data adapter via d_stage2."""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

from aibip import config, imgio
from aibip import d_data as data


def device_for(name='cpu'):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; select --device cpu')
    if name not in {'cpu', 'cuda'}:
        raise ValueError('device must be cpu, cuda or auto')
    return torch.device(name)


def losses(reconstruction, target, edge_weight=0.):
    import torch
    mse = torch.nn.functional.mse_loss(reconstruction, target)
    dx = torch.nn.functional.l1_loss(reconstruction[..., 1:]-reconstruction[..., :-1],
                                    target[..., 1:]-target[..., :-1])
    dy = torch.nn.functional.l1_loss(reconstruction[..., 1:, :]-reconstruction[..., :-1, :],
                                    target[..., 1:, :]-target[..., :-1, :])
    edge = (dx+dy)/2
    return mse+edge_weight*edge, mse, edge


def evaluate(model, loader, device, edge_weight):
    import torch
    model.eval()
    total = np.zeros(3, float)
    count = 0
    with torch.inference_mode():
        for x in loader:
            x = x.to(device)
            result = losses(model(x), x, edge_weight)
            values = np.array([float(v) for v in result])
            if not np.isfinite(values).all():
                raise RuntimeError('Non-finite validation loss')
            total += values*len(x)
            count += len(x)
    if not count:
        raise ValueError('Empty validation set')
    return (total/count).tolist()


def run_path(output_root, run_name, stage=1):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_name):
        raise ValueError('run-name must be a simple folder name')
    if stage not in (1, 2):
        raise ValueError('stage must be 1 or 2')
    return Path(output_root)/'runs'/run_name/f'stage{stage}'


def save_checkpoint(path, checkpoint):
    import torch
    temporary = path.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_best(output_root, run_name, device, validated=False, stage=1, data_module=data):
    data = data_module
    import torch
    from aibip.part_d import _build_model
    if not validated:
        data.validate(output_root)
    checkpoint = torch.load(run_path(output_root, run_name, stage)/'best.pt', map_location='cpu', weights_only=True)
    if checkpoint.get('stage') != stage:
        raise ValueError('Checkpoint stage differs from requested stage')
    if checkpoint['patch_manifest_sha256'] != data.sha(data.paths(output_root)[2]):
        raise ValueError('Model belongs to a different patch cache')
    model = _build_model().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def write_history(run_dir, history):
    if not history:
        return
    tmp = run_dir/'history.csv.tmp'
    with tmp.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    tmp.replace(run_dir/'history.csv')


def curve(run_dir, history):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    epochs = [r['epoch'] for r in history]
    for key, axis, title in [('mse', axes[0], 'MSE'), ('edge', axes[1], 'Gradient L1')]:
        axis.plot(epochs, [r['train_'+key] for r in history], label='train: noisy input')
        axis.plot(epochs, [r['val_'+key] for r in history], label='validation: clean input')
        axis.set(xlabel='Epoch', ylabel=title)
        axis.legend()
    best = min(history, key=lambda r: r['val_mse'])['epoch']
    axes[0].axvline(best, color='gray', linestyle=':', label=f'best {best}')
    fig.suptitle(f'D {run_dir.name} / {run_dir.parent.name} / best MSE epoch {best}')
    fig.savefig(run_dir/'loss_curve.png', dpi=140)
    plt.close(fig)


def train(stage=1, output_root=data.DEFAULT_OUTPUT, run_name='mse', epochs=30,
          device='cpu', batch_size=64, lr=.001, noise_std=.02, edge_weight=0.,
          patience=5, seed=42, threads=4, resume=False, data_module=data):
    data = data_module
    if stage != getattr(data, 'STAGE', 1):
        raise ValueError('Stage does not match data adapter; use aibip.d_stage2 for Stage 2')
    if stage == 2 and epochs > 30:
        raise ValueError('Stage 2 is limited to 30 total epochs')
    if min(epochs, batch_size, patience, threads) < 1 or not np.isfinite([lr, noise_std, edge_weight]).all():
        raise ValueError('Invalid training settings')
    if lr <= 0 or noise_std < 0 or edge_weight < 0:
        raise ValueError('Learning rate must be positive, loss/noise weights nonnegative')
    meta = data.validate(output_root)
    if meta.get('stage', stage) != stage:
        raise ValueError('Patch manifest belongs to a different stage')
    shard_path, index_path, manifest_path = data.paths(output_root)
    from aibip.part_d import _build_model, _make_dataset, read_index, select_rows
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    target_device = device_for(device)
    rows = read_index(index_path)
    train_rows = select_rows(rows, meta['split']['train_ids'])
    val_rows = select_rows(rows, meta['split']['val_ids'])
    run_dir = run_path(output_root, run_name, stage)
    settings = dict(batch_size=batch_size, lr=lr, noise_std=noise_std, edge_weight=edge_weight,
                    patience=patience, seed=seed, threads=threads, device=str(target_device),
                    model='ConvAE-32-64-128-256', selection_metric='val_mse',
                    training_source_sha256=data.sha(__file__))
    if not resume and ((run_dir/'best.pt').exists() or (run_dir/'last.pt').exists()):
        raise ValueError('Run already exists: use --resume or another --run-name')
    if resume and not (run_dir/'last.pt').exists():
        raise FileNotFoundError('Resume requires last.pt (optimizer, scheduler and RNG state)')
    run_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    model = _build_model().to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2)
    history, best_val, bad_epochs, start = [], math.inf, 0, 1
    manifest_hash = data.sha(manifest_path)
    if resume:
        checkpoint = torch.load(run_dir/'last.pt', map_location='cpu', weights_only=True)
        if (checkpoint.get('stage') != stage or checkpoint['settings'] != settings
                or checkpoint['patch_manifest_sha256'] != manifest_hash):
            raise ValueError('Resume settings, training code or patch cache differ from checkpoint')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Adam state must follow the model when resuming on CUDA.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != 'step':
                    state[key] = value.to(target_device)
        scheduler.load_state_dict(checkpoint['scheduler'])
        generator.set_state(checkpoint['loader_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if target_device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        history, best_val = checkpoint['history'], checkpoint['best_val']
        bad_epochs, start = checkpoint['bad_epochs'], checkpoint['epoch']+1
        if not (run_dir/'best.pt').exists():
            raise FileNotFoundError('Best checkpoint missing; cannot resume safely')
    train_loader = torch.utils.data.DataLoader(_make_dataset(shard_path, train_rows),
        batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = torch.utils.data.DataLoader(_make_dataset(shard_path, val_rows),
        batch_size=batch_size, shuffle=False, num_workers=0)
    for epoch in range(start, epochs+1):
        if bad_epochs >= patience:
            print('Early stopping patience reached; keeping the saved best model')
            break
        model.train()
        total, seen = np.zeros(3, float), 0
        for x in train_loader:
            x = x.to(target_device)
            noisy = (x+torch.randn_like(x)*noise_std).clamp(0, 1)
            objective, mse, edge = losses(model(noisy), x, edge_weight)
            if not torch.isfinite(objective):
                raise RuntimeError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += np.array([float(v.detach()) for v in (objective, mse, edge)])*len(x)
            seen += len(x)
        val_objective, val_mse, val_edge = evaluate(model, val_loader, target_device, edge_weight)
        train_objective, train_mse, train_edge = total/seen
        history.append(dict(epoch=epoch, train_objective=float(train_objective), train_mse=float(train_mse),
                            train_edge=float(train_edge), val_objective=val_objective,
                            val_mse=val_mse, val_edge=val_edge, lr=optimizer.param_groups[0]['lr']))
        improved = val_mse < best_val
        if improved:
            best_val, bad_epochs = val_mse, 0
        else:
            bad_epochs += 1
        scheduler.step(val_mse)
        common = dict(stage=stage, epoch=epoch, state_dict=model.state_dict(), settings=settings,
                      patch_manifest_sha256=manifest_hash, val_loss=val_mse,
                      train_ids=meta['split']['train_ids'], val_ids=meta['split']['val_ids'],
                      buffer_ids=meta['split']['buffer_ids'], color_order='BGR',
                      n_train_patches=len(train_rows), n_val_patches=len(val_rows))
        if improved:
            save_checkpoint(run_dir/'best.pt', common)
        save_checkpoint(run_dir/'last.pt', dict(common, optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), torch_rng=torch.get_rng_state(), loader_rng=generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if target_device.type == 'cuda' else [],
            history=history, best_val=best_val, bad_epochs=bad_epochs))
        write_history(run_dir, history)
        print(f'epoch {epoch:3}: train MSE {train_mse:.6f}, val MSE {val_mse:.6f}, '
              f'val edge {val_edge:.6f}' + (' *' if improved else ''), flush=True)
    if not history:
        raise ValueError('No training history; epochs must be positive')
    curve(run_dir, history)
    # Reload best.pt, not the last in-memory epoch, for all reported reconstructions.
    preview(stage, output_root, run_name, device=device, n=12, validated=True, data_module=data)
    print(f'D finished: best validation MSE {best_val:.6f}; {run_dir}')


def preview(stage=1, output_root=data.DEFAULT_OUTPUT, run_name='mse', device='cpu', n=12,
            seed=42, validated=False, data_module=data):
    data = data_module
    if stage != getattr(data, 'STAGE', 1):
        raise ValueError('Stage does not match data adapter')
    if n < 1:
        raise ValueError('n must be positive')
    import torch
    torch.set_num_threads(4)
    target_device = device_for(device)
    model, checkpoint = load_best(output_root, run_name, target_device, validated, stage, data)
    from aibip.part_d import read_index, _make_dataset, _reconstruct_patches
    shard_path, index_path, manifest_path = data.paths(output_root)
    meta = data.read_json(manifest_path)
    rows = read_index(index_path)
    shard = np.load(shard_path, mmap_mode='r')
    loader = torch.utils.data.DataLoader(_make_dataset(shard_path, rows), batch_size=64, shuffle=False)
    errors, edges = [], []
    with torch.inference_mode():
        for x in loader:
            x = x.to(target_device)
            reconstruction = model(x)
            errors.extend(((reconstruction-x)**2).mean((1,2,3)).cpu().tolist())
            dx = ((reconstruction[...,1:]-reconstruction[...,:-1])-(x[...,1:]-x[...,:-1])).abs().mean((1,2,3))
            dy = ((reconstruction[...,1:,:]-reconstruction[...,:-1,:])-(x[...,1:,:]-x[...,:-1,:])).abs().mean((1,2,3))
            edges.extend(((dx+dy)/2).cpu().tolist())
    errors, edges = np.array(errors), np.array(edges)
    if not np.isfinite(errors).all() or not np.isfinite(edges).all():
        raise RuntimeError('Non-finite preview error')
    membership = {key: set(checkpoint[key+'_ids']) for key in ['train', 'val', 'buffer']}
    label_for = lambda image_id: next((key for key, ids in membership.items() if image_id in ids), 'unseen')
    run_dir = run_path(output_root, run_name, stage)
    summary = {}
    for label in membership:
        take = np.array([r['image_id'] in membership[label] for r in rows])
        summary[label] = dict(n_patches=int(take.sum()), mse=float(errors[take].mean()) if take.any() else None,
                              edge=float(edges[take].mean()) if take.any() else None)
    data.save_json(run_dir/'metrics.json', dict(best_epoch=checkpoint['epoch'], metrics=summary,
                   patch_manifest_sha256=checkpoint['patch_manifest_sha256'], checkpoint_sha256=data.sha(run_dir/'best.pt')))
    by_image = {}
    for i, row in enumerate(rows):
        by_image.setdefault(row['image_id'], []).append(i)
    with (run_dir/'image_scores.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['image_id','split','n_patches','mean_error','max_error'])
        writer.writeheader()
        for src in meta['sources']:
            idx = by_image.get(src['image_id'], [])
            writer.writerow(dict(image_id=src['image_id'], split=label_for(src['image_id']), n_patches=len(idx),
                                 mean_error=float(errors[idx].mean()) if idx else None,
                                 max_error=float(errors[idx].max()) if idx else None))
    # Same seeded validation examples across all experiments; extremes separate.
    val_idx = np.array([i for i, row in enumerate(rows) if label_for(row['image_id']) == 'val'])
    picks = np.random.default_rng(seed).choice(val_idx, min(n, len(val_idx)), replace=False).tolist()
    def sheet(indices, filename):
        recon = _reconstruct_patches(model, shard, [rows[i]['patch_idx'] for i in indices], target_device)
        tiles = []
        for j, i in enumerate(indices):
            original = np.array(shard[rows[i]['patch_idx']])
            pixel_error = ((original.astype(np.float32)/255-recon[j])**2).mean(2)
            heat = cv2.applyColorMap(np.rint(np.clip(pixel_error/.01, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            tile = np.zeros((180, 384, 3), np.uint8)
            tile[52:] = np.hstack([original, np.rint(recon[j]*255).astype(np.uint8), heat])
            cv2.putText(tile, f"{rows[i]['image_id']} {label_for(rows[i]['image_id'])} b{rows[i]['band']}",
                        (4,16), cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cv2.putText(tile, f'MSE {errors[i]:.6f} | error scale 0-.01', (4,33), cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            tiles.append(tile)
        imgio.save_image(run_dir/filename, imgio.montage(tiles, cols=3))
    sheet(picks, 'validation_preview.jpg')
    count = min(6, len(rows))
    order = np.argsort(errors, kind='stable')
    sheet(order[:count].tolist(), 'best_patches.jpg')
    sheet(order[-count:][::-1].tolist(), 'worst_patches.jpg')
    data.save_json(run_dir/'preview_selection.json', dict(seed=seed, best_epoch=checkpoint['epoch'],
                   validation_patch_indices=[rows[i]['patch_idx'] for i in picks], error_scale=[0,.01]))
    print(f'D preview: best epoch {checkpoint["epoch"]}, val MSE {summary["val"]["mse"]:.6f}; {run_dir}')
    return summary


def compare(output_root=data.DEFAULT_OUTPUT, names=('mse', 'mse_edge')):
    """Compare saved metrics only; require identical patches/split and experiment controls."""
    import torch
    data.validate(output_root)
    records, reference = [], None
    for name in names:
        directory = run_path(output_root, name)
        ckpt = torch.load(directory/'best.pt', map_location='cpu', weights_only=True)
        if ckpt['patch_manifest_sha256'] != data.sha(data.paths(output_root)[2]):
            raise ValueError(f'{name}: model belongs to a different patch cache')
        metrics = data.read_json(directory/'metrics.json')
        if metrics['checkpoint_sha256'] != data.sha(directory/'best.pt'):
            raise ValueError(f'{name}: preview metrics are stale')
        controls = dict(ckpt['settings'])
        controls.pop('edge_weight')
        signature = (ckpt['patch_manifest_sha256'], controls)
        if reference is not None and signature != reference:
            raise ValueError('Runs differ in data or controls besides edge_weight; not a controlled loss comparison')
        reference = signature
        records.append(dict(run=name, epoch=ckpt['epoch'], edge_weight=ckpt['settings']['edge_weight'],
                            **metrics['metrics']['val']))
    data.save_json(Path(output_root)/'comparison.json', records)
    for record in records:
        print(f"{record['run']}: best epoch {record['epoch']}, val MSE {record['mse']:.6f}, edge {record['edge']:.6f}")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description='D v2: auditable Mini experiments')
    sub = parser.add_subparsers(dest='command', required=True)
    extract_parser = sub.add_parser('extract')
    train_parser = sub.add_parser('train')
    preview_parser = sub.add_parser('preview')
    compare_parser = sub.add_parser('compare')
    for p in [extract_parser, train_parser, preview_parser, compare_parser]:
        p.add_argument('--output-root', type=Path, default=data.DEFAULT_OUTPUT)
    for p in [extract_parser, train_parser, preview_parser]:
        p.add_argument('--stage', type=int, choices=[1], default=1)
    extract_parser.add_argument('--metadata', type=Path, default=data.DEFAULT_METADATA)
    extract_parser.add_argument('--cap', type=int, default=24)
    extract_parser.add_argument('--spacing', type=int, default=64)
    extract_parser.add_argument('--filter', dest='filter_mode', choices=['green','none','legacy'], default='green')
    extract_parser.add_argument('--appearance', choices=['clahe','original'], default='clahe')
    extract_parser.add_argument('--cell-m', type=float, default=300.)
    extract_parser.add_argument('--buffer-m', type=float, default=30.)
    extract_parser.add_argument('--val-fraction', type=float, default=.2)
    extract_parser.add_argument('--seed', type=int, default=42)
    for p in [train_parser, preview_parser]:
        p.add_argument('--run-name', default='mse')
        p.add_argument('--device', choices=['cpu','cuda','auto'], default='cpu')
        p.add_argument('--seed', type=int, default=42)
    train_parser.add_argument('--epochs', type=int, default=30)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--lr', type=float, default=.001)
    train_parser.add_argument('--noise-std', type=float, default=.02)
    train_parser.add_argument('--edge-weight', type=float, default=0.)
    train_parser.add_argument('--patience', type=int, default=5)
    train_parser.add_argument('--threads', type=int, default=4)
    train_parser.add_argument('--resume', action='store_true')
    preview_parser.add_argument('--n', type=int, default=12)
    compare_parser.add_argument('--runs', nargs=2, default=['mse','mse_edge'])
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    if command == 'extract':
        data.extract(**args)
    elif command == 'train':
        train(**args)
    elif command == 'preview':
        preview(**args)
    else:
        compare(args['output_root'], args['runs'])
