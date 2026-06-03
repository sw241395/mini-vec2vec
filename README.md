# mini-vec2vec

This is a python package implementation of the [mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations](https://arxiv.org/abs/2510.02348) paper.

This is **not** the official python implementation that is used in the paper, that code can be found [here](https://github.com/guy-dar/mini-vec2vec). All credit goes to `guy-dar`, this just takes his work but puts it into a package. 

## Install

```bash
pip install mini-vec2vec
```

## Contribute

Create a venv and install the dev dependencies. (I have used UV to create this package)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv sync
uvx pre-commit install
```

### Testing

```bash
uv pip install -e .
uv run pytest
```

### Docs

Docs are created using [zensical](https://github.com/zensical/zensical), to start the docs locally:

```bash
uv run zensical serve
```

--- 

## TODO:

* Code Improvements:
    * Make some of the hard coded values variables
* Performance Improvements:
    * Use float32 numpy arrays
    * Use `from scipy.optimize import linear_sum_assignment` rather than QAP or reduce number of QAP loops
    * `MiniBatchKMeans` instead of `Kmeans`
        * Could we expand to any clustering algorithm?
    * Concatenate rather than vstack
* Add in original Vec2Vec method
* Add example for hyper parameter tuning using grid search or bayesian optimization 
* Rewrite docstring in the [Google Style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) to work effectively with mkdocstrings
