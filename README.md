## Installation

Code is a fork of OSRL and you need to install osrl first:

```bash
pip install osrl-lib
```

### Training
To train the `pdca` method, simply run by overriding the default parameters:

```shell
python examples/train/train_pdca.py --task OfflineCarCircle-v0 --param1 args1 ...
```

### Evaluation
To evaluate a trained agent, for example, a pdca agent, simply run
```shell
python examples/eval/eval_pdca.py --path path_to_model --eval_episodes 20
```
