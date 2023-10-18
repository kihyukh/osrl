## Installation

Code is a fork of OSRL. To install OSRL:

```bash
pip install osrl-lib
```

You can also pull the repo and install:
```bash
git clone https://github.com/liuzuxin/OSRL.git
cd osrl
pip install -e .
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
