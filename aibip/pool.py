"""Clone-ready pool preparation and training, without Mini data dependencies.

No command chains automatically into segmentation or training.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from aibip import config, imgio, pool_data, d_training
from aibip.d_stage2 import write_csv


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    parsers = {name: sub.add_parser(name) for name in ['inventory','segment','extract','train','preview']}
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


if __name__ == '__main__':
    main()
