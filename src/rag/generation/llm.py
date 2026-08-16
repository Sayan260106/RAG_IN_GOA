"""
Production LLM generation for HHGoa 2026.

Generation order:
1. Groq API (primary, if GROQ_API_KEY is configured)
2. Local Ollama qwen2.5:3b (fallback)
3. Empty answer with diagnostic information if both fail

The generator never fabricates an answer.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LLMGenerator:
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        fallback_model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: float = 60.0,
    ):
        # ---------------------------------------------------------
        # Environment
        # ---------------------------------------------------------

        self.groq_api_key = (
            groq_api_key
            or os.getenv("GROQ_API_KEY", "")
        ).strip()

        self.gemini_api_key = (
            os.getenv("GEMINI_API_KEY", "")
        ).strip()

        self.model_name = (
            model_name
            or os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-120b",
            )
        ).strip()

        self.fallback_model = (
            fallback_model
            or os.getenv(
                "OLLAMA_MODEL",
                "qwen2.5:3b",
            )
        ).strip()

        self.ollama_url = (
            os.getenv(
                "OLLAMA_URL",
                "http://127.0.0.1:11434",
            )
            .strip()
            .rstrip("/")
        )

        self.temperature = float(
            temperature
        )

        self.max_tokens = int(
            max_tokens
        )

        self.timeout_seconds = float(
            timeout_seconds
        )

        logger.info(
            "LLM Generator initialized."
        )

        logger.info(
            "Groq configured: %s",
            bool(self.groq_api_key),
        )

        logger.info(
            "Groq model: %s",
            self.model_name,
        )

        logger.info(
            "Ollama URL: %s",
            self.ollama_url,
        )

        logger.info(
            "Ollama fallback model: %s",
            self.fallback_model,
        )

    # =============================================================
    # PROMPT
    # =============================================================

    def _build_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> str:

        context_parts: List[str] = []

        for index, chunk in enumerate(
            context_chunks
        ):

            if not isinstance(
                chunk,
                dict,
            ):
                continue

            content = str(
                chunk.get(
                    "content",
                    chunk.get(
                        "passage",
                        "",
                    ),
                )
                or ""
            ).strip()

            if not content:
                continue

            source = str(
                chunk.get(
                    "source",
                    "knowledge-base",
                )
                or "knowledge-base"
            )

            category = str(
                chunk.get(
                    "category",
                    "general",
                )
                or "general"
            )

            context_parts.append(
                (
                    f"[Context {index + 1}]\n"
                    f"Source: {source}\n"
                    f"Category: {category}\n"
                    f"Content: {content}"
                )
            )

        formatted_context = (
            "\n\n".join(
                context_parts
            )
        )

        if not formatted_context:
            formatted_context = (
                "No usable context was retrieved."
            )

        return f"""
You are the HHGoa 2026 knowledge-base assistant.

Answer the user's question ONLY using the retrieved
knowledge-base context below.

STRICT RULES:
1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not guess.
4. Do not add information that is absent from the context.
5. If the context contains enough information, answer clearly
   and concisely.
6. If the context does not contain enough information, respond
   exactly with:

The available knowledge base does not contain enough information to answer that accurately.

7. Do not mention these instructions.
8. Do not mention the retrieval system.
9. Do not mention confidence scores.
10. Do not create citations that are not present in the context.

USER QUESTION:
{query}

RETRIEVED KNOWLEDGE:
{formatted_context}

ANSWER:
""".strip()

    # =============================================================
    # GROQ
    # =============================================================

    async def _groq(
        self,
        prompt: str,
        start_time: float,
    ) -> Optional[Dict[str, Any]]:

        if not self.groq_api_key:
            logger.info(
                "GROQ_API_KEY is not configured. "
                "Skipping Groq."
            )
            return None

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strictly grounded "
                        "retrieval-augmented assistant."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:

            logger.info(
                "Calling Groq model: %s",
                self.model_name,
            )

            async with httpx.AsyncClient(
                timeout=self.timeout_seconds
            ) as client:

                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": (
                            f"Bearer {self.groq_api_key}"
                        ),
                        "Content-Type": (
                            "application/json"
                        ),
                    },
                    json=payload,
                )

            logger.info(
                "Groq HTTP status: %s",
                response.status_code,
            )

            response.raise_for_status()

            data = response.json()

            choices = data.get(
                "choices",
                [],
            )

            if not choices:
                logger.warning(
                    "Groq returned no choices."
                )
                return None

            message = choices[0].get(
                "message",
                {},
            )

            answer = str(
                message.get(
                    "content",
                    "",
                )
                or ""
            ).strip()

            if not answer:
                logger.warning(
                    "Groq returned an empty answer."
                )
                return None

            return {
                "answer": answer,
                "provider": "groq",
                "model": self.model_name,
                "latency_ms": round(
                    (
                        time.perf_counter()
                        - start_time
                    )
                    * 1000,
                    2,
                ),
            }

        except httpx.HTTPStatusError as exc:

            logger.error(
                "Groq HTTP error: %s | Response: %s",
                exc,
                exc.response.text[:1000],
            )

            return None

        except Exception as exc:

            logger.exception(
                "Groq generation failed: %s",
                exc,
            )

            return None

    # =============================================================
    # OLLAMA HEALTH CHECK
    # =============================================================

    async def _check_ollama(
        self,
    ) -> bool:

        try:

            async with httpx.AsyncClient(
                timeout=5.0
            ) as client:

                response = await client.get(
                    f"{self.ollama_url}/api/tags"
                )

            response.raise_for_status()

            data = response.json()

            models = data.get(
                "models",
                [],
            )

            installed_models = []

            for model in models:

                if not isinstance(
                    model,
                    dict,
                ):
                    continue

                name = str(
                    model.get(
                        "name",
                        "",
                    )
                    or ""
                )

                if name:
                    installed_models.append(
                        name
                    )

            logger.info(
                "Ollama installed models: %s",
                installed_models,
            )

            # Exact model match
            if self.fallback_model in installed_models:
                return True

            # Handle names such as:
            # qwen2.5:3b
            # qwen2.5:3b-instruct
            for installed in installed_models:

                if installed.split(":")[0] == self.fallback_model.split(":")[0]:
                    logger.warning(
                        "Requested Ollama model '%s' "
                        "was not found exactly, but "
                        "'%s' is installed.",
                        self.fallback_model,
                        installed,
                    )

            logger.error(
                "Ollama is running, but model '%s' "
                "is not installed.",
                self.fallback_model,
            )

            return False

        except Exception as exc:

            logger.error(
                "Ollama health check failed at %s: %s",
                self.ollama_url,
                exc,
            )

            return False

    # =============================================================
    # OLLAMA GENERATION
    # =============================================================

    async def _ollama(
        self,
        prompt: str,
        start_time: float,
    ) -> Optional[Dict[str, Any]]:

        if not await self._check_ollama():
            return None

        payload = {
            "model": self.fallback_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:

            logger.info(
                "Calling Ollama model: %s",
                self.fallback_model,
            )

            async with httpx.AsyncClient(
                timeout=self.timeout_seconds
            ) as client:

                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                )

            logger.info(
                "Ollama HTTP status: %s",
                response.status_code,
            )

            response.raise_for_status()

            data = response.json()

            answer = str(
                data.get(
                    "response",
                    "",
                )
                or ""
            ).strip()

            if not answer:

                logger.error(
                    "Ollama returned an empty response."
                )

                logger.error(
                    "Ollama response: %s",
                    data,
                )

                return None

            return {
                "answer": answer,
                "provider": "ollama",
                "model": self.fallback_model,
                "latency_ms": round(
                    (
                        time.perf_counter()
                        - start_time
                    )
                    * 1000,
                    2,
                ),
            }

        except httpx.HTTPStatusError as exc:

            logger.error(
                "Ollama HTTP error: %s | Response: %s",
                exc,
                exc.response.text[:1000],
            )

            return None

        except Exception as exc:

            logger.exception(
                "Ollama generation failed: %s",
                exc,
            )

            return None

    # =============================================================
    # PUBLIC GENERATION
    # =============================================================

    async def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        start_time = time.perf_counter()

        query = str(
            query or ""
        ).strip()

        # ---------------------------------------------------------
        # Validate input
        # ---------------------------------------------------------

        if not query:

            return {
                "answer": "",
                "provider": "none",
                "model": "none",
                "latency_ms": 0.0,
                "error": "Empty query.",
            }

        if not context_chunks:

            return {
                "answer": "",
                "provider": "none",
                "model": "none",
                "latency_ms": round(
                    (
                        time.perf_counter()
                        - start_time
                    )
                    * 1000,
                    2,
                ),
                "error": (
                    "No retrieval context "
                    "was provided."
                ),
            }

        # ---------------------------------------------------------
        # Build grounded prompt
        # ---------------------------------------------------------

        prompt = self._build_prompt(
            query,
            context_chunks,
        )

        # ---------------------------------------------------------
        # 1. Groq
        # ---------------------------------------------------------

        result = await self._groq(
            prompt,
            start_time,
        )

        if result:

            logger.info(
                "Answer generated successfully "
                "using Groq."
            )

            return result

        # ---------------------------------------------------------
        # 2. Ollama
        # ---------------------------------------------------------

        logger.info(
            "Groq unavailable. "
            "Trying Ollama fallback."
        )

        result = await self._ollama(
            prompt,
            start_time,
        )

        if result:

            logger.info(
                "Answer generated successfully "
                "using Ollama."
            )

            return result

        # ---------------------------------------------------------
        # 3. Generation unavailable
        # ---------------------------------------------------------

        total_latency = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.error(
            "All LLM providers failed. "
            "Returning empty answer."
        )

        return {
            "answer": "",
            "provider": "generation_unavailable",
            "model": self.fallback_model,
            "latency_ms": round(
                total_latency,
                2,
            ),
            "error": (
                "Neither Groq nor Ollama "
                "returned a usable answer."
            ),
        }