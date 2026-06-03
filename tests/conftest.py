import pytest
import numpy
from pathlib import Path
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


@pytest.fixture(scope="function")
def A():
    embeddings_file_path = Path("./data/A.npy")
    if embeddings_file_path.exists():
        embeddings = numpy.load(embeddings_file_path)
    else:
        # Embed sentences with a small embedding model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode(SENTENCES, convert_to_numpy=True)
        embeddings_file_path.parent.mkdir(parents=True, exist_ok=True)
        numpy.save(embeddings_file_path, embeddings)
    return embeddings


@pytest.fixture(scope="function")
def B():
    embeddings_file_path = Path("./data/B.npy")
    if embeddings_file_path.exists():
        embeddings = numpy.load(embeddings_file_path)
    else:
        # Embed sentences with a small embedding model
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        embeddings = model.encode(SENTENCES, convert_to_numpy=True)
        embeddings_file_path.parent.mkdir(parents=True, exist_ok=True)
        numpy.save(embeddings_file_path, embeddings)
    return embeddings
