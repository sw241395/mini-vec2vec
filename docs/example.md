# Example notebook for using Mini-Vec2vec

This is a recreation of the original notebook created by `guy-dar` but using the newly created python package.

Once again all credit goes to `guy-dar` and if you want more information about how it work I would recommend going to his [GitHub page](https://github.com/guy-dar/mini-vec2vec).


```python
# Install the package
! pip install mini-vec2vec
# or local install
# ! pip install -e ..
```


```python
# Set up test data
# I will be reusing the data `guy-dar` has created in his code
from huggingface_hub import hf_hub_download
import torch

# Load in embeddings
repo_id = "dar-tau/nq-embeddings"
filenames = {"e5": "e5.pt", "gtr": "gtr.pt"}
sent_embeds = {
    key: torch.load(
        hf_hub_download(repo_id=repo_id, filename=filename), map_location="cpu"
    )
    for key, filename in filenames.items()
}
embed_A, embed_B = sent_embeds["e5"].cpu(), sent_embeds["gtr"].cpu()

# Get subset of the data and convert to numpy arrays
n = len(embed_A) - 8192

X_train = embed_A[: n // 2].numpy()
Y_train = embed_B[n // 2 : n].numpy()

X_eval = embed_A[n:].numpy()
Y_eval = embed_B[n:].numpy()
```

## Basic example

Here is a basic demo of how to use the package to create a linear mapping from embedding space `A` to embedding space `B`.


```python
import mini_vec2vec

# Create MiniVec2Vec object 
mv2v = mini_vec2vec.MiniVec2Vec()
# If you are wanting to use the centroid only method from the paper
# mv2v = mini_vec2vec.CentroidMiniVec2Vec

# Fit transform from A to B
mv2v.fit(X_train, Y_train)
```


```python
# Transform embeddings
transformed_X_eval = mv2v.transform(X_eval)
```


```python
# Evaluation
# I have used the same eval method `guy-dar` used in his example
from sklearn.metrics.pairwise import cosine_similarity


def rank(X):
    return torch.argsort(torch.argsort(X, dim=-1), dim=-1)


similarities = cosine_similarity(transformed_X_eval, Y_eval)

ranks = rank(torch.tensor(similarities)).diagonal()

print("Top-1 Accuracy:", (len(X_eval) - 1 == ranks).float().mean())
print("Average Rank:", len(X_eval) - ranks.float().mean())
# Top-1 Accuracy: tensor(0.8824)
# Average Rank: tensor(1.4531)
```

## More control example

The `fit` method in the example above is good for if you want a quick and easy generalized method.

You are able to have more control over each step in the paper for a more fine grained training:
* `match_anchors`: Step 1 of the mini-vec2vec method where you are clustering and then starting an initial transform matrix `w` using procrustes
* `refinement_1`: Refinement 1 in the paper
* `refinement_2`: Refinement 1 in the paper

**Note:** Running `fit(X_train, Y_train)` is the same as running:

```python
mv2v.match_anchors(X_train, Y_train).refinement_1(X_train, Y_train).refinement_2(X_train, Y_train)
```

but running each step explicitly, you can set the arguments of each step explicitly.


```python
# Create new object
mv2v = mini_vec2vec.MiniVec2Vec()

mv2v.match_anchors(X_train, Y_train)
mv2v.refinement_1(X_train, Y_train)
mv2v.refinement_2(X_train, Y_train)

# Transform embeddings just like before
transformed_X_eval = mv2v.transform(X_eval)
```

You can find more details about the package API in the [documentation](https://sw241395.github.io/mini-vec2vec/)


