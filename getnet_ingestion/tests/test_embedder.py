import json
import tempfile
import unittest
from pathlib import Path

from getnet_ingestion.embedder import (
    EMBEDDING_DIMENSIONS,
    EmbeddingError,
    MODEL_NAME,
    PASSAGE_PREFIX,
    load_records,
    vectorize_records,
    write_records,
)


class FakeTokenizer:
    def __call__(self, text, *, add_special_tokens, truncation, verbose):
        self.truncation = truncation
        tokens = text.split()
        if add_special_tokens:
            tokens = ["[CLS]", *tokens, "[SEP]"]
        return {"input_ids": tokens}


class FakeModel:
    max_seq_length = 8

    def __init__(self, dimensions=EMBEDDING_DIMENSIONS):
        self.dimensions = dimensions
        self.tokenizer = FakeTokenizer()
        self.encoded_texts = None

    def get_embedding_dimension(self):
        return self.dimensions

    def encode(self, texts, **kwargs):
        self.encoded_texts = texts
        return [[1.0] + [0.0] * (self.dimensions - 1) for _ in texts]


class VectorizeRecordsTests(unittest.TestCase):
    def test_preserves_record_and_adds_normalized_embedding_metadata(self):
        record = {"id": "section-1", "text": "A few words", "category": "solution"}
        model = FakeModel()

        result = vectorize_records([record], model, "test-model")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], record["id"])
        self.assertEqual(result[0]["category"], record["category"])
        self.assertEqual(result[0]["embedding_model"], "test-model")
        self.assertEqual(result[0]["embedding_dimensions"], EMBEDDING_DIMENSIONS)
        self.assertEqual(len(result[0]["embedding"]), EMBEDDING_DIMENSIONS)
        self.assertEqual(model.encoded_texts, [f"{PASSAGE_PREFIX}{record['text']}"])
        self.assertEqual(MODEL_NAME, "intfloat/multilingual-e5-small")
        self.assertFalse(model.tokenizer.truncation)

    def test_separates_oversized_sections_at_semantic_block_boundaries(self):
        model = FakeModel()
        model.max_seq_length = 10
        record = {
            "id": "too-long",
            "section_heading": "Page title - Section heading",
            "text": "Page title - Section heading\nFirst paragraph.\nSecond paragraph.\nThird paragraph.",
        }

        result = vectorize_records([record], model)

        self.assertEqual(len(result), 3)
        self.assertEqual(
            [item["text"].splitlines()[-1] for item in result],
            ["First paragraph.", "Second paragraph.", "Third paragraph."],
        )
        self.assertEqual(
            [item["section_heading"] for item in result],
            [record["section_heading"]] * 3,
        )
        self.assertEqual(len({item["id"] for item in result}), 3)
        self.assertTrue(all(text.startswith(PASSAGE_PREFIX) for text in model.encoded_texts))
        self.assertTrue(
            all(not {"parent_id", "chunk_index", "chunk_count"}.intersection(item) for item in result)
        )
        self.assertEqual(len(model.encoded_texts), 3)

    def test_splits_oversized_paragraphs_at_sentence_boundaries_without_loss(self):
        model = FakeModel()
        model.max_seq_length = 14
        sentences = (
            "First sentence has four words.",
            "Second sentence has five words.",
            "Third sentence has five words.",
        )
        record = {
            "id": "long-paragraph",
            "section_heading": "Title - Section",
            "text": f"Title - Section\n{' '.join(sentences)}",
        }

        result = vectorize_records([record], model)

        self.assertGreater(len(result), 1)
        combined_text = " ".join(item["text"] for item in result)
        for sentence in sentences:
            self.assertEqual(combined_text.count(sentence), 1)
        self.assertTrue(all(text.startswith(PASSAGE_PREFIX) for text in model.encoded_texts))

    def test_does_not_split_a_single_oversized_semantic_block(self):
        model = FakeModel()
        record = {
            "id": "too-long-block",
            "section_heading": "Title - Section",
            "text": "Title - Section\nThis single paragraph is too long.",
        }

        with self.assertRaisesRegex(EmbeddingError, "not divided or truncated"):
            vectorize_records([record], model)

        self.assertIsNone(model.encoded_texts)

    def test_refuses_unexpected_embedding_dimension(self):
        with self.assertRaisesRegex(EmbeddingError, "Expected 384-dimensional"):
            vectorize_records([{"id": "section-1", "text": "A few words"}], FakeModel(3))


class JsonlTests(unittest.TestCase):
    def test_loads_records_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "corpus.jsonl"
            path.write_text(
                '{"id":"one","text":"First"}\n{"id":"one","text":"Second"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EmbeddingError, "Duplicate record id"):
                load_records(path)

    def test_writes_jsonl_records(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nested" / "embeddings.jsonl"
            records = [{"id": "one", "text": "Texto", "embedding": [0.1, 0.2]}]

            write_records(records, path)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), records[0])


if __name__ == "__main__":
    unittest.main()
