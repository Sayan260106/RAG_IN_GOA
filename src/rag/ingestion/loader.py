"""
Streaming dataset loader for HHGoa 2026.

Primary production corpus:
    ai4bharat/MSMARCO-XI

The loader uses Hugging Face streaming=True and reads only a bounded
number of examples. It never downloads the complete dataset.
"""

import logging
import os
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_CONFIGS = {
    "as", "bn", "gu", "hi", "kn", "ml", "mr",
    "ne", "or", "pa", "sa", "ta", "te", "ur",
}


class MSMARCOStreamingLoader:
    def __init__(
        self,
        dataset_name: str = "ai4bharat/MSMARCO-XI",
        split: str = "train",
        languages: Optional[List[str]] = None,
        max_samples: int = 500,
        samples_per_language: int = 50,
    ):
        self.dataset_name = os.getenv(
            "HF_DATASET_NAME", dataset_name
        ).strip()
        self.split = os.getenv(
            "HF_DATASET_SPLIT", split
        ).strip()

        env_languages = os.getenv(
            "HF_LANGUAGES",
            "hi,bn,gu,kn,ml,mr,or,ta,te,ur",
        )

        self.languages = [
            x.strip().lower()
            for x in (
                languages
                if languages is not None
                else env_languages.split(",")
            )
            if x.strip()
        ]

        self.max_samples = int(
            os.getenv("HF_MAX_SAMPLES", str(max_samples))
        )
        self.samples_per_language = int(
            os.getenv(
                "HF_SAMPLES_PER_LANGUAGE",
                str(samples_per_language),
            )
        )

        self.use_hf_dataset = (
            os.getenv("USE_HF_DATASET", "true").strip().lower()
            == "true"
        )
        self.allow_local_fallback = (
            os.getenv("ALLOW_LOCAL_FALLBACK", "false").strip().lower()
            == "true"
        )

    def stream_dataset(self) -> Iterator[Dict[str, Any]]:
        if not self.use_hf_dataset:
            logger.info(
                "USE_HF_DATASET=false. Using built-in fallback corpus."
            )
            yield from self._get_fallback_corpus()
            return

        try:
            yield from self._stream_huggingface()
        except Exception as exc:
            if not self.allow_local_fallback:
                logger.exception(
                    "Hugging Face streaming failed and local fallback is disabled."
                )
                raise RuntimeError(
                    "Could not stream MSMARCO-XI. Check internet access, "
                    "dataset config, and HF credentials if required."
                ) from exc

            logger.warning(
                "Hugging Face streaming failed (%s). "
                "Using local fallback corpus because "
                "ALLOW_LOCAL_FALLBACK=true.",
                exc,
            )
            yield from self._get_fallback_corpus()

    def _stream_huggingface(self) -> Iterator[Dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The 'datasets' package is not installed. "
                "Install it with: pip install datasets"
            ) from exc

        if not self.languages:
            raise ValueError(
                "HF_LANGUAGES is empty. Configure at least one "
                "MSMARCO-XI language."
            )

        invalid = [
            language
            for language in self.languages
            if language not in SUPPORTED_CONFIGS
        ]
        if invalid:
            raise ValueError(
                "Unsupported MSMARCO-XI language config(s): "
                + ", ".join(invalid)
                + ". Supported configs: "
                + ", ".join(sorted(SUPPORTED_CONFIGS))
            )

        logger.info(
            "Streaming %s/%s. Target=%d rows, %d per language. "
            "The complete dataset will NOT be downloaded.",
            self.dataset_name,
            self.split,
            self.max_samples,
            self.samples_per_language,
        )

        total = 0

        for language in self.languages:
            if total >= self.max_samples:
                break

            remaining = self.max_samples - total
            language_limit = min(
                self.samples_per_language,
                remaining,
            )

            logger.info(
                "Opening streaming config '%s' (limit=%d).",
                language,
                language_limit,
            )

            # IMPORTANT: MSMARCO-XI has language configs such as "hi",
            # "bn", "gu", etc. There is no single generic "en" config.
            dataset = load_dataset(
                self.dataset_name,
                language,
                split=self.split,
                streaming=True,
            )

            language_count = 0

            for item in dataset:
                if (
                    language_count >= language_limit
                    or total >= self.max_samples
                ):
                    break

                if not isinstance(item, dict):
                    continue

                document = self._normalize_hf_item(
                    item=item,
                    index=total,
                    language_config=language,
                )

                if document is None:
                    continue

                yield document
                total += 1
                language_count += 1

            logger.info(
                "Streamed %d usable examples from config '%s'.",
                language_count,
                language,
            )

        logger.info(
            "Finished bounded MSMARCO-XI streaming: %d documents.",
            total,
        )

    def _normalize_hf_item(
        self,
        item: Dict[str, Any],
        index: int,
        language_config: str,
    ) -> Optional[Dict[str, Any]]:
        query = self._first_text(
            item,
            ["query", "Eng_Query"],
        )

        passage = self._extract_passage(item)

        if not passage:
            return None

        document_id = self._first_text(
            item,
            ["query_id", "id", "doc_id", "document_id"],
        ) or f"msmarco-xi-{language_config}-{index}"

        target_lang = self._first_text(
            item,
            ["target_lang", "language", "lang"],
        ) or language_config

        language = target_lang.split("_", 1)[0].lower()

        metadata = {
            "dataset": self.dataset_name,
            "config": language_config,
            "split": self.split,
            "stream_index": index,
            "target_lang": target_lang,
        }

        for key in (
            "source_lang",
            "target_lang",
            "query_id",
            "query_type",
            "Eng_Query",
            "Eng_Answer",
        ):
            if key in item:
                metadata[key] = item[key]

        return {
            "id": str(document_id),
            "query": str(query or ""),
            "passage": str(passage).strip(),
            "language": language,
            "source": self.dataset_name,
            "metadata": metadata,
        }

    def _extract_passage(
        self,
        item: Dict[str, Any],
    ) -> str:
        """
        MSMARCO-XI stores:
            passages = {
                "is_selected": [...],
                "English_passages": [...],
                "Translated_passages": [...]
            }
        """
        for key in (
            "passage",
            "text",
            "content",
            "document",
            "context",
        ):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        passages = item.get("passages")

        if isinstance(passages, dict):
            translated = passages.get(
                "Translated_passages"
            ) or []
            english = passages.get(
                "English_passages"
            ) or []
            selected = passages.get(
                "is_selected"
            ) or []

            if isinstance(translated, list):
                selected_translated = [
                    translated[i]
                    for i in range(len(translated))
                    if (
                        i < len(selected)
                        and selected[i] == 1
                        and isinstance(translated[i], str)
                        and translated[i].strip()
                    )
                ]
                if selected_translated:
                    return selected_translated[0].strip()

                for value in translated:
                    if isinstance(value, str) and value.strip():
                        return value.strip()

            if isinstance(english, list):
                selected_english = [
                    english[i]
                    for i in range(len(english))
                    if (
                        i < len(selected)
                        and selected[i] == 1
                        and isinstance(english[i], str)
                        and english[i].strip()
                    )
                ]
                if selected_english:
                    return selected_english[0].strip()

                for value in english:
                    if isinstance(value, str) and value.strip():
                        return value.strip()

        return ""

    @staticmethod
    def _first_text(
        item: Dict[str, Any],
        keys: List[str],
    ) -> str:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue

            if isinstance(value, str) and value.strip():
                return value.strip()

            if isinstance(value, (int, float)):
                return str(value)

        return ""

    def _get_fallback_corpus(self) -> List[Dict[str, Any]]:
        """
        Development-only emergency fallback.

        The original local corpus is preserved in
        local_fallback_corpus.py.
        """
        fallback_file = os.path.join(
            os.path.dirname(__file__),
            "local_fallback_corpus.py",
        )

        if not os.path.exists(fallback_file):
            logger.warning(
                "No local fallback corpus found."
            )
            return []

        namespace: Dict[str, Any] = {}
        with open(fallback_file, "r", encoding="utf-8") as handle:
            exec(handle.read(), namespace)

        corpus = namespace.get("FALLBACK_CORPUS", [])
        return corpus if isinstance(corpus, list) else []
