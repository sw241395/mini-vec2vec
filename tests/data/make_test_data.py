"""
Here is a basic Python script to create the test data used for
the test in thisnrepo. I decided on the models:

* jinaai/jina-embeddings-v5-text-small
* Qwen/Qwen3-Embedding-0.6B

becasue they are Matryoshka embeddings meaning I can truncate down
to 32 dims to save storage.

To recreate the embeddings, run:
```
cd ./tests/data/
uv run python make_test_data.py
```
"""

import numpy
from sentence_transformers import SentenceTransformer

# 16 GPT generated random sentences

SENTENCES = [
    "The lighthouse continued to flash through the fog long after the fishing boats had returned to shore.",
    "A small cactus can survive for weeks without rain by storing water in its thick stem.",
    "The violinist practiced the same passage repeatedly until every note sounded effortless.",
    "Several satellites orbit Earth to provide navigation, communication, and weather data.",
    "The bakery filled the street with the smell of fresh bread before sunrise.",
    "Ancient civilizations often built monuments that still attract visitors thousands of years later.",
    "A curious fox paused at the edge of the forest and watched the hikers pass by.",
    "The software update reduced loading times and improved overall system stability.",
    "During autumn, many tree species shed their leaves as daylight hours become shorter.",
    "The chef added a squeeze of lemon to brighten the flavor of the soup.",
    "Researchers are studying coral reefs to better understand how marine ecosystems respond to change.",
    "A well-designed bicycle can travel efficiently using only human power.",
    "The museum displayed artifacts recovered from a shipwreck discovered off the coast.",
    "Learning a new language often becomes easier with regular conversation and practice.",
    "Thunder echoed across the valley moments after a bright flash of lightning.",
    "The chess player sacrificed a rook to create a winning position several moves later.",
]

model = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-small", trust_remote_code=True
)
embeddings = model.encode(
    SENTENCES,
    convert_to_numpy=True,
    truncate_dim=32,
    task="text-matching",
)
numpy.savetxt("A.out", embeddings, delimiter=",")

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
embeddings = model.encode(SENTENCES, convert_to_numpy=True, truncate_dim=32)
numpy.savetxt("B.out", embeddings, delimiter=",")
