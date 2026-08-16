import numpy as np

from src.rag.embeddings.model import MultilingualEmbeddingModel


def test_embedding_model_initialization():
    model = MultilingualEmbeddingModel(
        model_name="BAAI/bge-m3"
    )

    assert model.model_name == "BAAI/bge-m3"
    assert model.dimension == 1024


def test_query_embedding_dimension():
    model = MultilingualEmbeddingModel(
        model_name="BAAI/bge-m3"
    )

    embedding = model.encode_query(
        "What are the best places to visit in Goa?"
    )

    assert embedding.shape == (1024,)
    assert embedding.dtype == np.float32


def test_passage_embedding_dimension():
    model = MultilingualEmbeddingModel(
        model_name="BAAI/bge-m3"
    )

    embedding = model.encode_passage(
        "Goa is known for its beaches, churches, "
        "heritage sites and Portuguese architecture."
    )

    assert embedding.shape == (1024,)
    assert embedding.dtype == np.float32


def test_embedding_is_normalized():
    model = MultilingualEmbeddingModel(
        model_name="BAAI/bge-m3"
    )

    embedding = model.encode_query(
        "Tell me about Goa tourism."
    )

    norm = np.linalg.norm(embedding)

    assert np.isclose(
        norm,
        1.0,
        atol=1e-3,
    )


def test_batch_embedding_dimension():
    model = MultilingualEmbeddingModel(
        model_name="BAAI/bge-m3"
    )

    texts = [
        "Goa has beautiful beaches.",
        "Goa has many historical churches.",
        "Goa is a popular tourist destination.",
    ]

    embeddings = model.encode_passages(texts)

    assert embeddings.shape == (
        3,
        1024,
    )

    assert embeddings.dtype == np.float32


def test_empty_batch_embedding():
    model = MultilingualEmbeddingModel(
        model_name="BAAI/bge-m3"
    )

    embeddings = model.encode_passages([])

    assert embeddings.shape == (
        0,
        1024,
    )