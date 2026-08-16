"""
Robust Dataset Loader for HHGoa Voice RAG.

Primary source:
    ai4bharat/MSMARCO-XI

Behavior:
    - Uses Hugging Face streaming only when USE_HF_DATASET=true.
    - Never downloads the complete dataset.
    - Falls back to the built-in Goa/Indic corpus if HF is unavailable.
    - Always returns a consistent document structure.
"""

import logging
import os
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class MSMARCOStreamingLoader:
    """
    Streaming loader for MSMARCO-XI with a reliable local fallback.

    Returned document format:

    {
        "id": str,
        "query": str,
        "passage": str,
        "language": str,
        "source": str,
        "metadata": dict
    }
    """

    def __init__(
        self,
        dataset_name: str = "ai4bharat/MSMARCO-XI",
        split: str = "train",
        languages: Optional[List[str]] = None,
        max_samples: int = 1000,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.languages = languages or ["en", "hi", "kok"]
        self.max_samples = max_samples

        self.use_hf_dataset = (
            os.getenv("USE_HF_DATASET", "false").strip().lower()
            == "true"
        )
        self.allow_local_fallback = (
            os.getenv("ALLOW_LOCAL_FALLBACK", "true").strip().lower()
            == "true"
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def stream_dataset(self) -> Iterator[Dict[str, Any]]:
        """
        Stream documents from Hugging Face or fallback corpus.
        """

        if not self.use_hf_dataset:
            logger.info(
                "USE_HF_DATASET=false. "
                "Using built-in Goa & Indic knowledge corpus."
            )

            yield from self._get_fallback_corpus()
            return

        try:
            yield from self._stream_huggingface()

        except Exception as exc:
            if not self.allow_local_fallback:
                logger.exception(
                    "Hugging Face dataset is enabled but could not be loaded."
                )
                raise RuntimeError(
                    "USE_HF_DATASET=true but the Hugging Face corpus could not "
                    "be loaded. Set ALLOW_LOCAL_FALLBACK=true only for development."
                ) from exc

            logger.warning(
                "Hugging Face dataset failed (%s). Using local fallback corpus "
                "because ALLOW_LOCAL_FALLBACK=true.",
                exc,
            )
            yield from self._get_fallback_corpus()

    # ============================================================
    # HUGGING FACE STREAMING
    # ============================================================

    def _stream_huggingface(
        self,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream MSMARCO-XI without downloading the entire dataset.
        """

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The 'datasets' package is not installed. "
                "Install it with: pip install datasets"
            ) from exc

        logger.info(
            "Connecting to Hugging Face dataset '%s' "
            "using streaming mode.",
            self.dataset_name,
        )

        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
        )

        logger.info(
            "Hugging Face features: %s",
            getattr(dataset, "features", "unknown"),
        )

        count = 0

        for item in dataset:

            if count >= self.max_samples:
                break

            if not isinstance(item, dict):
                continue

            document = self._normalize_hf_item(
                item=item,
                index=count,
            )

            if document is None:
                continue

            # Optional language filtering.
            language = document.get("language", "en")

            if (
                self.languages
                and language not in self.languages
            ):
                continue

            yield document

            count += 1

        logger.info(
            "Finished streaming %d documents from %s.",
            count,
            self.dataset_name,
        )

    # ============================================================
    # NORMALIZE HF RECORD
    # ============================================================

    def _normalize_hf_item(
        self,
        item: Dict[str, Any],
        index: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert different possible MSMARCO-XI record formats
        into the internal document schema.
        """

        # --------------------------------------------------------
        # Query
        # --------------------------------------------------------

        query = self._first_text(
            item,
            [
                "query",
                "question",
                "title",
            ],
        )

        # --------------------------------------------------------
        # Passage / document text
        # --------------------------------------------------------

        passage = self._extract_passage(item)

        if not passage:
            return None

        # --------------------------------------------------------
        # ID
        # --------------------------------------------------------

        document_id = self._first_text(
            item,
            [
                "id",
                "query_id",
                "doc_id",
                "document_id",
            ],
        )

        if not document_id:
            document_id = f"msmarco-xi-{index}"

        # --------------------------------------------------------
        # Language
        # --------------------------------------------------------

        language = self._first_text(
            item,
            [
                "lang",
                "language",
                "language_code",
            ],
        )

        if not language:
            language = "en"

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        metadata = {
            "dataset": self.dataset_name,
            "split": self.split,
            "stream_index": index,
        }

        # Preserve useful dataset metadata.
        for key in (
            "lang",
            "language",
            "query_id",
            "doc_id",
            "source",
        ):
            if key in item:
                metadata[key] = item[key]

        return {
            "id": str(document_id),
            "query": str(query or ""),
            "passage": str(passage).strip(),
            "language": str(language),
            "source": self.dataset_name,
            "metadata": metadata,
        }

    # ============================================================
    # PASSAGE EXTRACTION
    # ============================================================

    def _extract_passage(
        self,
        item: Dict[str, Any],
    ) -> str:
        """
        Extract passage text from several possible dataset layouts.
        """

        # Direct text fields.
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

        # Positive passages.
        positive = item.get("positive_passages")

        if isinstance(positive, list):
            for entry in positive:

                if isinstance(entry, str):
                    if entry.strip():
                        return entry.strip()

                if isinstance(entry, dict):
                    for key in (
                        "text",
                        "passage",
                        "content",
                        "document",
                    ):
                        value = entry.get(key)

                        if (
                            isinstance(value, str)
                            and value.strip()
                        ):
                            return value.strip()

        # Generic passages list.
        passages = item.get("passages")

        if isinstance(passages, list):
            for entry in passages:

                if isinstance(entry, str):
                    if entry.strip():
                        return entry.strip()

                if isinstance(entry, dict):
                    for key in (
                        "text",
                        "passage",
                        "content",
                    ):
                        value = entry.get(key)

                        if (
                            isinstance(value, str)
                            and value.strip()
                        ):
                            return value.strip()

        return ""

    # ============================================================
    # SAFE FIELD EXTRACTION
    # ============================================================

    @staticmethod
    def _first_text(
        item: Dict[str, Any],
        keys: List[str],
    ) -> str:
        """
        Return the first non-empty textual field.
        """

        for key in keys:
            value = item.get(key)

            if value is None:
                continue

            if isinstance(value, str):
                if value.strip():
                    return value.strip()

            elif isinstance(value, (int, float)):
                return str(value)

        return ""

    # ============================================================
    # FALLBACK CORPUS
    # ============================================================

    def _get_fallback_corpus(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Built-in Goa knowledge corpus.

        This guarantees that the RAG system can run without:
            - Hugging Face
            - datasets package
            - internet access
            - HF authentication
        """

        return [

            # ----------------------------------------------------
            # 1. Goa Climate
            # ----------------------------------------------------

            {
                "id": "goa-met-001",
                "query": (
                    "What are the primary factors affecting "
                    "monsoon patterns in North Goa?"
                ),
                "passage": (
                    "In North Goa, monsoon patterns are primarily "
                    "governed by the South-West monsoon winds "
                    "originating over the Arabian Sea. The Western "
                    "Ghats (Sahyadri range) create an orographic "
                    "barrier, forcing moisture-laden oceanic air to "
                    "rise and condense rapidly, yielding heavy "
                    "rainfall across coastal and mid-hinterland "
                    "areas between June and September."
                ),
                "language": "en",
                "source": "Goa Meteorological Climate Records",
                "metadata": {
                    "category": "meteorology",
                    "region": "North Goa",
                    "type": "climate",
                },
            },

            # ----------------------------------------------------
            # 2. Basilica of Bom Jesus
            # ----------------------------------------------------

            {
                "id": "goa-her-002",
                "query": (
                    "What is the architectural and historical "
                    "significance of the Basilica of Bom Jesus?"
                ),
                "passage": (
                    "The Basilica of Bom Jesus in Old Goa is a "
                    "landmark of Baroque architecture in India. "
                    "The basilica is associated with the relics "
                    "of St. Francis Xavier and forms part of the "
                    "Churches and Convents of Goa UNESCO World "
                    "Heritage complex."
                ),
                "language": "en",
                "source": "Goa Heritage Records",
                "metadata": {
                    "category": "heritage",
                    "region": "Old Goa",
                    "type": "history",
                },
            },

            # ----------------------------------------------------
            # 3. Goan Fish Curry
            # ----------------------------------------------------

            {
                "id": "goa-cui-003",
                "query": (
                    "What spices and ingredients are essential "
                    "for authentic Goan Fish Curry?"
                ),
                "passage": (
                    "Authentic Goan Fish Curry, commonly called "
                    "Xitt Codi, uses coconut, red chillies, "
                    "coriander, cumin and turmeric. Kokum is a "
                    "traditional souring ingredient. Fish such "
                    "as kingfish or pomfret may be used depending "
                    "on the recipe and local availability."
                ),
                "language": "en",
                "source": "Goan Culinary Knowledge Records",
                "metadata": {
                    "category": "cuisine",
                    "region": "Goa",
                    "type": "food",
                },
            },

            # ----------------------------------------------------
            # 4. Dudhsagar Falls
            # ----------------------------------------------------

            {
                "id": "goa-eco-004",
                "query": (
                    "How can I visit Dudhsagar Falls and "
                    "what is the best season?"
                ),
                "passage": (
                    "Dudhsagar Falls is a major waterfall located "
                    "within the Bhagwan Mahavir Wildlife Sanctuary "
                    "on the Goa-Karnataka border. Access can depend "
                    "on forest regulations and seasonal conditions. "
                    "The post-monsoon and winter period is generally "
                    "more suitable for visiting, while heavy monsoon "
                    "conditions can restrict access."
                ),
                "language": "en",
                "source": "Goa Eco Tourism Knowledge Records",
                "metadata": {
                    "category": "ecotourism",
                    "region": "South Goa",
                    "type": "wildlife",
                },
            },

            # ----------------------------------------------------
            # 5. Olive Ridley Turtles
            # ----------------------------------------------------

            {
                "id": "goa-wild-005",
                "query": (
                    "Where do Olive Ridley turtles nest in Goa "
                    "and how are they protected?"
                ),
                "passage": (
                    "Olive Ridley sea turtles nest on several "
                    "beaches along the Goa coast. Important nesting "
                    "areas include Galgibaga, Agonda, Morjim and "
                    "Mandrem. Conservation measures include "
                    "protected nesting areas, hatcheries, monitoring "
                    "and restrictions intended to reduce disturbance "
                    "to nesting turtles and hatchlings."
                ),
                "language": "en",
                "source": "Goa Marine Wildlife Records",
                "metadata": {
                    "category": "conservation",
                    "region": "Goa Coast",
                    "type": "marine_life",
                },
            },

            # ----------------------------------------------------
            # 6. Fontainhas
            # ----------------------------------------------------

            {
                "id": "goa-her-006",
                "query": (
                    "What makes Fontainhas Latin Quarter unique?"
                ),
                "passage": (
                    "Fontainhas is the Latin Quarter of Panaji and "
                    "is known for its narrow streets, colourful "
                    "Portuguese-style houses, balconies and "
                    "distinctive architectural character. The area "
                    "preserves an important part of Goa's Indo-"
                    "Portuguese urban heritage."
                ),
                "language": "en",
                "source": "Panaji Heritage Records",
                "metadata": {
                    "category": "heritage",
                    "region": "Panaji",
                    "type": "architecture",
                },
            },

            # ----------------------------------------------------
            # 7. Goa Beaches
            # ----------------------------------------------------

            {
                "id": "goa-beach-007",
                "query": (
                    "What are some famous beaches in Goa?"
                ),
                "passage": (
                    "Goa has numerous beaches along its Arabian "
                    "Sea coastline. Popular beaches include "
                    "Baga, Calangute, Anjuna, Vagator, Palolem, "
                    "Agonda and Morjim. Different beaches are known "
                    "for different experiences including nightlife, "
                    "relaxation, water activities and scenic beauty."
                ),
                "language": "en",
                "source": "Goa Tourism Knowledge Records",
                "metadata": {
                    "category": "tourism",
                    "region": "Goa",
                    "type": "beaches",
                },
            },

            # ----------------------------------------------------
            # 8. Goa Feni
            # ----------------------------------------------------

            {
                "id": "goa-food-008",
                "query": (
                    "What is Feni and how is it traditionally made?"
                ),
                "passage": (
                    "Feni is a traditional spirit associated with "
                    "Goa. Cashew feni is produced from the juice "
                    "of ripe cashew apples through fermentation "
                    "and distillation. Coconut feni is traditionally "
                    "associated with fermented coconut palm sap. "
                    "Traditional production methods form an important "
                    "part of Goa's culinary and cultural heritage."
                ),
                "language": "en",
                "source": "Goan Culinary Heritage Records",
                "metadata": {
                    "category": "culture",
                    "region": "Goa",
                    "type": "traditional_beverage",
                },
            },

            # ----------------------------------------------------
            # 9. Goa History
            # ----------------------------------------------------

            {
                "id": "goa-his-009",
                "query": (
                    "What is the history of Portuguese rule in Goa?"
                ),
                "passage": (
                    "Portuguese rule in Goa began in the early "
                    "sixteenth century after the capture of Goa "
                    "in 1510. Portuguese influence continued for "
                    "several centuries and significantly shaped "
                    "Goan architecture, religion, cuisine, language "
                    "and cultural traditions. Goa became part of "
                    "India in 1961."
                ),
                "language": "en",
                "source": "Goa Historical Knowledge Records",
                "metadata": {
                    "category": "history",
                    "region": "Goa",
                    "type": "colonial_history",
                },
            },

            # ----------------------------------------------------
            # 10. Goa Culture
            # ----------------------------------------------------

            {
                "id": "goa-culture-010",
                "query": (
                    "What are important cultural traditions of Goa?"
                ),
                "passage": (
                    "Goan culture reflects a combination of "
                    "Konkani traditions and centuries of cultural "
                    "interaction with Portuguese and other communities. "
                    "Music, festivals, cuisine, traditional houses, "
                    "local crafts and community celebrations are "
                    "important elements of Goan cultural identity."
                ),
                "language": "en",
                "source": "Goa Cultural Knowledge Records",
                "metadata": {
                    "category": "culture",
                    "region": "Goa",
                    "type": "traditions",
                },
            },
        ]