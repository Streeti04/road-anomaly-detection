"""CPU smoke/resume test in a subprocess; parent test suite stays torch-free."""
import subprocess
import sys
from pathlib import Path


def test_cpu_training_resume_and_checkpoint_roundtrip(tmp_path):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(tmp_path)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr


def worker(root):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from aibip import part_d, d_data as data, d_training as training
    torch.set_num_threads(1)
    torch.manual_seed(42)
    # Actual production model: finite forward/backward and an observable update.
    model = part_d._build_model()
    x = torch.rand(2, 3, 128, 128)
    initial = next(model.parameters()).detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    prediction = model(x)
    assert prediction.shape == x.shape and torch.isfinite(prediction).all()
    objective, mse, edge = training.losses(prediction, x, .1)
    assert torch.isfinite(objective) and objective >= mse
    objective.backward()
    optimizer.step()
    assert not torch.equal(initial, next(model.parameters()))
    assert training.losses(x, x, .1)[0].item() == 0
    # Tiny model keeps repeated training/resume tests inexpensive. The production
    # training loop, Adam/scheduler, RNG restoration and checkpoint IO are real.
    part_d._build_model = lambda: torch.nn.Sequential(torch.nn.Conv2d(3,3,1), torch.nn.Sigmoid())
    shard, index, manifest = data.paths(root)
    shard.parent.mkdir(parents=True)
    np.save(shard, np.random.default_rng(4).integers(30,190,(8,128,128,3),dtype=np.uint8))
    rows = [dict(patch_idx=i,image_id='a' if i < 4 else 'b',band=1,row=100,col=100) for i in range(8)]
    part_d.write_index(index, rows)
    meta = dict(split=dict(train_ids=['a'],val_ids=['b'],buffer_ids=[]))
    data.save_json(manifest, meta)
    data.validate = lambda path: meta
    training.preview = lambda *a, **kw: None
    training.curve = lambda *a: None
    settings = dict(output_root=root, device='cpu', batch_size=2, threads=1, patience=10)
    training.train(run_name='continuous', epochs=2, **settings)
    training.train(run_name='resumed', epochs=1, **settings)
    training.train(run_name='resumed', epochs=2, resume=True, **settings)
    load = lambda name, filename: torch.load(training.run_path(root,name)/filename, weights_only=True)
    continuous, resumed = load('continuous','last.pt'), load('resumed','last.pt')
    assert continuous['history'] == resumed['history']
    assert all(torch.equal(value,resumed['state_dict'][name]) for name,value in continuous['state_dict'].items())
    restored, checkpoint = training.load_best(root, 'resumed', torch.device('cpu'), validated=True)
    again, _ = training.load_best(root, 'resumed', torch.device('cpu'), validated=True)
    restored.eval()
    again.eval()
    assert torch.equal(restored(x), again(x))
    try:
        training.train(run_name='resumed', epochs=3, resume=True, edge_weight=.1, **settings)
    except ValueError:
        pass
    else:
        raise AssertionError('Resume must reject changed loss settings')
    print('Production model backward pass and deterministic epoch-boundary resume verified')


if __name__ == '__main__':
    worker(Path(sys.argv[2]))
