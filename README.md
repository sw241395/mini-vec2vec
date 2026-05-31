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

--- 

## TODO:

* Tests
* Docs ([zensical](https://github.com/zensical/zensical))
* Code Improvements:
    * Separate out the 3 main parts in fit to make them able to run as separate steps for even more control
    * Make some of the hard coded values variables
    * Test zero padding for different dimension embeddings
* Performance Improvements:
    * Use float32 numpy arrays
    * Use `from scipy.optimize import linear_sum_assignment` rather than QAP or reduce number of QAP loops
    * `MiniBatchKMeans` instead of `Kmeans`
    * Concatenate rather than vstack
* Add no-relative representation variant
* Add in original Vec2Vec method
* Gitlab CI to auto run tests?
