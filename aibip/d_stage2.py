"""Explicit pool preparation, training, and Mini evaluation with the Stage-2 model.

No command chains automatically into segmentation or training.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from aibip import config, imgio, d_data, d_pool, d_training


def write_csv(path, rows, fields):
    temporary = path.with_suffix('.tmp.csv')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def ranked_images(image_ids, rows, errors, split):
    """Median MSE is the ranking signal; low-support images stay ranked but flagged."""
    errors = np.asarray(errors, dtype=float)
    if len(errors) != len(rows) or not np.isfinite(errors).all():
        raise ValueError('Expected one finite reconstruction error per patch')
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['image_id'], []).append(i)
    membership = {k: set(split[k+'_ids']) for k in ['train', 'val', 'buffer']}
    records = []
    for key in sorted(image_ids):
        indices = groups.get(key, [])
        count = len(indices)
        records.append(dict(image_id=key, split=next((k for k, ids in membership.items() if key in ids), 'unseen'),
                            n_patches=count, median_error=float(np.median(errors[indices])) if indices else None,
                            mean_error=float(errors[indices].mean()) if indices else None,
                            max_error=float(errors[indices].max()) if indices else None,
                            rank_typical_to_anomalous=None,
                            status='scored' if count >= 3 else ('low_support' if count else 'no_patches')))
    scored = sorted((r for r in records if r['n_patches']), key=lambda r: (r['median_error'], r['image_id']))
    for rank, record in enumerate(scored, 1):
        record['rank_typical_to_anomalous'] = rank
    return scored+[r for r in records if not r['n_patches']], groups


def analyze(output_root=d_pool.DEFAULT_OUTPUT, run_name='mse', device='cpu', n=6):
    """Assignment E(a) and D Mini reconstructions; manual interpretation remains E(b-d)."""
    if n < 1 or n > 24:
        raise ValueError('n must be between 1 and 24')
    from aibip.part_d import read_index, _make_dataset, _reconstruct_patches
    import torch
    torch.set_num_threads(4)
    meta = d_pool.validate(output_root)
    # Recheck the current Mini set, including byte identity to the pool.
    info = d_pool.inventory(meta['pool_dir'], meta['metadata'], meta['inventory']['mini_dir'])
    if info['mini_ids'] != meta['inventory']['mini_ids']:
        raise ValueError('Mini membership changed since extraction')
    target = d_training.device_for(device)
    model, checkpoint = d_training.load_best(output_root, run_name, target, True, 2, d_pool)
    shard_path, index_path, _ = d_pool.paths(output_root)
    mini_ids = set(info['mini_ids'])
    rows = [r for r in read_index(index_path) if r['image_id'] in mini_ids]
    errors = []
    if rows:
        loader = torch.utils.data.DataLoader(_make_dataset(shard_path, rows), batch_size=64, shuffle=False)
        with torch.inference_mode():
            for x in loader:
                x = x.to(target)
                errors.extend(((model(x)-x)**2).mean((1,2,3)).cpu().tolist())
    errors = np.asarray(errors)
    ranking, by_image = ranked_images(mini_ids, rows, errors, meta['split'])
    dest = d_training.run_path(output_root, run_name, 2)/'mini_analysis'
    dest.mkdir(parents=True, exist_ok=True)
    fields = ['image_id','split','n_patches','median_error','mean_error','max_error','rank_typical_to_anomalous','status']
    write_csv(dest/'ranking.csv', ranking, fields)
    patch_records = [dict(**row, mse=float(errors[i])) for i, row in enumerate(rows)]
    write_csv(dest/'patch_scores.csv', patch_records, ['patch_idx','image_id','band','row','col','mse'])
    # Keep the user's annotations across reruns: never overwrite this file.
    review = dest/'manual_review.csv'
    if not review.exists():
        write_csv(review, [dict(image_id=r['image_id'], observation='', possible_confounder='', comment='')
                           for r in ranking], ['image_id','observation','possible_confounder','comment'])
    shard = np.load(shard_path, mmap_mode='r')
    rank_by_id = {r['image_id']: r for r in ranking}
    def sheet(indices, filename):
        if not indices:
            return
        recon = _reconstruct_patches(model, shard, [rows[i]['patch_idx'] for i in indices], target)
        tiles = []
        for j, i in enumerate(indices):
            row = rows[i]
            image = rank_by_id[row['image_id']]
            original = np.array(shard[row['patch_idx']])
            pixel_error = ((original.astype(np.float32)/255-recon[j])**2).mean(2)
            heat = cv2.applyColorMap(np.rint(np.clip(pixel_error/.01, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            tile = np.zeros((194,384,3), np.uint8)
            tile[66:] = np.hstack([original, np.rint(recon[j]*255).astype(np.uint8), heat])
            lines = [f"{row['image_id']} {image['split']} b{row['band']}",
                     f"image median {image['median_error']:.6f} | patch {errors[i]:.6f}",
                     'Original | Reconstruction | MSE 0-.01']
            for y, text in zip([15,33,51], lines):
                cv2.putText(tile, text, (4,y), cv2.FONT_HERSHEY_SIMPLEX, .38, (255,255,255), 1)
            tiles.append(tile)
        imgio.save_image(dest/filename, imgio.montage(tiles, cols=3))
    scored = [r for r in ranking if r['n_patches']]
    typical, anomalous = scored[:n], scored[-n:][::-1]
    # Same representative rule at both ends: maximum-error patch of each image.
    representative = lambda records: [max(by_image[r['image_id']], key=lambda i: errors[i]) for r in records]
    typical_idx, anomalous_idx = representative(typical), representative(anomalous)
    sheet(typical_idx, 'most_typical.jpg')
    sheet(anomalous_idx, 'most_anomalous.jpg')
    order = np.argsort(errors, kind='stable')
    sheet(order[:n].tolist(), 'mini_best_patches.jpg')
    sheet(order[-n:][::-1].tolist(), 'mini_worst_patches.jpg')
    d_pool.save_json(dest/'analysis.json', dict(stage=2, best_epoch=checkpoint['epoch'],
        checkpoint_sha256=d_pool.sha(dest.parent/'best.pt'),
        patch_manifest_sha256=checkpoint['patch_manifest_sha256'], n_images=len(ranking),
        n_scored=len(scored), n_patches=len(rows), condition_signal='median patch RGB-channel MSE (BGR storage)',
        error_scale=[0,.01], representative='maximum-error patch per ranked image',
        typical_patch_indices=[rows[i]['patch_idx'] for i in typical_idx],
        anomalous_patch_indices=[rows[i]['patch_idx'] for i in anomalous_idx],
        interpretation='Visual unusualness, not a damage probability; Mini may overlap training'))
    print(f'E: Stage-2 model, {len(scored)}/{len(ranking)} Mini images ranked; {dest}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    parsers = {name: sub.add_parser(name) for name in ['inventory','segment','extract','train','preview','analyze']}
    for name, p in parsers.items():
        p.add_argument('--output-root', type=Path, default=d_pool.DEFAULT_OUTPUT)
        if name in {'inventory','segment','extract'}:
            p.add_argument('--pool-dir', type=Path, default=d_pool.DEFAULT_POOL)
            p.add_argument('--metadata', type=Path, default=d_data.DEFAULT_METADATA)
        if name in {'train','preview','analyze'}:
            p.add_argument('--run-name', default='mse')
            p.add_argument('--device', choices=['cpu','cuda','auto'], default='cpu')
    parsers['extract'].add_argument('--mini-reference', type=Path, default=d_data.DEFAULT_OUTPUT)
    p = parsers['train']
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--noise-std', type=float, default=.02)
    p.add_argument('--edge-weight', type=float, default=0.)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    parsers['preview'].add_argument('--n', type=int, default=12)
    parsers['analyze'].add_argument('--n', type=int, default=6)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    if Path.cwd().resolve() != d_pool.ROOT.resolve():
        parser.error('Run from the project root (C uses project-relative paths)')
    d_pool.safe_root(args['output_root'])
    if command in {'inventory', 'segment'}:
        info = d_pool.inventory(args['pool_dir'], args['metadata'])
        if command == 'inventory':
            import torch
            info['torch_version'] = str(torch.__version__)
            info['cuda_available'] = torch.cuda.is_available()
            info['cuda_device'] = torch.cuda.get_device_name(0) if info['cuda_available'] else None
            d_pool.save_json(Path(args['output_root'])/'inventory.json', info)
            print(f"Pool: {info['n_images']} files; Mini: {info['n_mini']}; "
                  f"metadata without files: {len(info['missing_image_ids'])}; CUDA: {info['cuda_available']}")
        else:
            from aibip.c_transfer import run
            # Exact existing C logic. Valid Mini masks are reused; conflicts stop.
            run(args['pool_dir'], backend=config.C_BACKEND, force=False)
    elif command == 'extract':
        d_pool.extract(**args)
    elif command == 'train':
        d_training.train(stage=2, data_module=d_pool, **args)
    elif command == 'preview':
        d_training.preview(stage=2, data_module=d_pool, **args)
    else:
        analyze(**args)


if __name__ == '__main__':
    main()
