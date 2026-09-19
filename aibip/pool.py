"""Clone-ready pool preparation and training, without Mini data dependencies.

No command chains automatically into segmentation or training.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from aibip import config, imgio, pool_data, d_training
from aibip.d_stage2 import ranked_images, write_csv


def evaluation_ids(meta, sample_dir=None):
    sources = {r['image_id']: r for r in meta['sources']}
    if sample_dir is None:
        return set(sources)
    sample = pool_data.find_images(sample_dir)
    if not sample:
        raise ValueError('The evaluation sample directory contains no images')
    for path in sample:
        if path.stem not in sources or pool_data.sha(path) != sources[path.stem]['source_sha256']:
            raise ValueError(f'{path.name}: sample must contain unchanged images from this pool')
    return {p.stem for p in sample}


def analyze(output_root=pool_data.DEFAULT_OUTPUT, run_name='mse', device='cpu', n=6, sample_dir=None):
    """Rank the pool or an optional subset; this is separate from training."""
    if n < 1 or n > 24:
        raise ValueError('n must be between 1 and 24')
    from aibip.part_d import read_index, _make_dataset, _reconstruct_patches
    import torch
    torch.set_num_threads(4)
    meta = pool_data.validate(output_root)
    selected_ids = evaluation_ids(meta, sample_dir)
    target = d_training.device_for(device)
    model, checkpoint = d_training.load_best(output_root, run_name, target, True, 2, pool_data)
    shard_path, index_path, _ = pool_data.paths(output_root)
    rows = [r for r in read_index(index_path) if r['image_id'] in selected_ids]
    errors = []
    if rows:
        loader = torch.utils.data.DataLoader(_make_dataset(shard_path, rows), batch_size=64, shuffle=False)
        with torch.inference_mode():
            for x in loader:
                x = x.to(target)
                errors.extend(((model(x)-x)**2).mean((1,2,3)).cpu().tolist())
    errors = np.asarray(errors)
    ranking, by_image = ranked_images(selected_ids, rows, errors, meta['split'])
    dest = d_training.run_path(output_root, run_name, 2)/('sample_analysis' if sample_dir is not None else 'pool_analysis')
    selection = dict(image_ids=sorted(selected_ids), sample_dir=str(Path(sample_dir).resolve()) if sample_dir else None)
    if (dest/'selection.json').exists() and pool_data.read_json(dest/'selection.json') != selection:
        raise ValueError('This analysis folder already belongs to another sample; retain it before changing the sample')
    dest.mkdir(parents=True, exist_ok=True)
    pool_data.save_json(dest/'selection.json', selection)
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
    sheet(order[:n].tolist(), 'best_patches.jpg')
    sheet(order[-n:][::-1].tolist(), 'worst_patches.jpg')
    pool_data.save_json(dest/'analysis.json', dict(stage=2, best_epoch=checkpoint['epoch'],
        checkpoint_sha256=pool_data.sha(dest.parent/'best.pt'),
        patch_manifest_sha256=checkpoint['patch_manifest_sha256'], n_images=len(ranking),
        n_scored=len(scored), n_patches=len(rows), condition_signal='median patch RGB-channel MSE (BGR storage)',
        error_scale=[0,.01], representative='maximum-error patch per ranked image',
        typical_patch_indices=[rows[i]['patch_idx'] for i in typical_idx],
        anomalous_patch_indices=[rows[i]['patch_idx'] for i in anomalous_idx],
        interpretation='Visual unusualness, not a damage probability; sample may overlap training'))
    print(f'E: Stage-2 model, {len(scored)}/{len(ranking)} images ranked; {dest}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    parsers = {name: sub.add_parser(name) for name in ['inventory','segment','extract','train','preview','analyze']}
    for name, p in parsers.items():
        p.add_argument('--output-root', type=Path, default=pool_data.DEFAULT_OUTPUT)
        if name in {'inventory','segment','extract'}:
            p.add_argument('--images', dest='pool_dir', type=Path, required=True)
            p.add_argument('--metadata', type=Path, required=True)
        if name in {'train','preview','analyze'}:
            p.add_argument('--run-name', default='mse')
            p.add_argument('--device', choices=['cpu','cuda','auto'], default='auto')
    parsers['segment'].add_argument('--limit', type=int)
    parsers['segment'].add_argument('--force', action='store_true',
                                    help='back up and replace incompatible existing masks')
    p = parsers['extract']
    p.add_argument('--cap', type=int, default=24)
    p.add_argument('--spacing', type=int, default=64)
    p.add_argument('--cell-m', type=float, default=300.)
    p.add_argument('--buffer-m', type=float, default=30.)
    p.add_argument('--val-fraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--appearance', choices=['clahe','original'], default='clahe')
    p.add_argument('--filter', dest='filter_mode', choices=['green','none','legacy'], default='green')
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
    parsers['analyze'].add_argument('--sample-dir', type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    if Path.cwd().resolve() != pool_data.ROOT.resolve():
        parser.error('Run from the cloned project root')
    root = pool_data.safe_root(args['output_root'])
    if command in {'inventory','segment'}:
        pool_path = args['pool_dir'].resolve()
        if root == pool_path or pool_path in root.parents:
            parser.error('Keep output-root outside the recursively scanned image directory')
        info = pool_data.inventory(pool_path, args['metadata'])
        if command == 'inventory':
            import torch
            from aibip.c_transfer import _run_key
            info['torch_version'] = str(torch.__version__)
            info['cuda_available'] = torch.cuda.is_available()
            info['cuda_device'] = torch.cuda.get_device_name(0) if info['cuda_available'] else None
            _, model_settings = _run_key(config.C_BACKEND)  # verify local weights; no model inference
            info['segmentation_settings'] = model_settings
            pool_data.save_json(root/'inventory.json', info)
            print(f"Pool: {info['n_images']} images; missing image files: {len(info['missing_image_ids'])}; "
                  f"CUDA: {info['cuda_available']}; no Mini files required")
        else:
            from aibip.c_transfer import _run_paths
            if args['limit'] is not None and args['limit'] < 1:
                parser.error('limit must be positive')
            _run_paths(pool_data.find_images(pool_path)[:args['limit']], backend=config.C_BACKEND, force=args['force'])
    elif command == 'extract':
        pool_data.extract(**args)
    elif command == 'train':
        d_training.train(stage=2, data_module=pool_data, **args)
    elif command == 'preview':
        d_training.preview(stage=2, data_module=pool_data, **args)
    else:
        analyze(**args)


if __name__ == '__main__':
    main()
