# mini-vec2vec

This is a python package implementation of the [mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations](https://github.com/guy-dar/mini-vec2vec/tree/main) paper.

All credit goes to the `guy-dar`, this just takes his work but puts it into an easy to install package.

---

## Install

```bash
pip install mini-vec2vec
```

---

## Contribute

Create a venv and install the dev dependencies. (I have used UV to create this package)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv sync
$ uvx pre-commit install
```

--- 

## TODO:

* Tests
* Docs ([zensical](https://github.com/zensical/zensical))
* Code Improvements:
    * Make some of the hard coded values variables
* Performance Improvements:
    * Use float32 numpy arrays
    * Use `from scipy.optimize import linear_sum_assignment` rather than QAP or reduce number of QAP loops
    * `MiniBatchKMeans` instead of `Kmeans`
    * Concatenate rather than vstack
* Add in original Vec2Vec method