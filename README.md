# CovXNet pneumonia classification

Builds a residual/dilated depthwise-convolution network for normal, viral, and bacterial pneumonia arrays.

## Setup and execution

Use Python 3.11 and a separate virtual environment for this project. Run commands from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_inputs.py
python -m jupyterlab
```

On Windows, activate with `.venv\Scripts\activate`. For notebooks, select this environment’s kernel and execute from the top in a fresh kernel. The first code cell exposes `INPUT_PATHS` where old machine-specific paths existed. `PROJECT_DATA_DIR` changes the base for those configured paths; remaining relative paths use the notebook working directory.

## Notebooks

- [covXnet_30_epoch_pneumonia.ipynb](covXnet_30_epoch_pneumonia.ipynb)

## Inputs

Supply these inputs yourself; no dataset is downloaded automatically:

- `train_Normal_128.npy`
- `train_Virus_128.npy`
- `train_bacteria_128.npy`
- `test_Normal_128.npy`
- `test_Virus_128.npy`
- `test_bacteria_128.npy`

## Outputs

Keras model/checkpoint files, training curves, and multiclass confusion matrices.

## Implementation and limitations

Supply numeric, non-pickled image arrays with dimensions expected by the notebook (128×128). covxnet.py holds the original residual/shifter architecture. Graph visualization additionally requires Graphviz. This experiment monitors test data each epoch; final unbiased evaluation needs a separate holdout.

Reusable model/analysis functions live in `experiment_covxnet_30_epoch_pneumonia.py`. Notebooks retain the experiment sequence, training, and visual narrative.

## Verification

The workspace regression suite checks source/notebook validity and selected behavior with bundled or synthetic inputs. Full experiment training and historical metric reproduction are not implied. See the workspace `REFACTOR_REPORT.md` for exact test coverage and unresolved runtime requirements.

## Original references

These links were retained from the original project notes; availability and licensing must be checked by the person obtaining the data.

- https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
