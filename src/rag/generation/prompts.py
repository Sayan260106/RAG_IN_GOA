from typing import List, Dict, Any

class PromptBuilder:
    def build_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        formatted_context = "\n\n".join([f"[{i+1}] {c.get('content', '')}" for i, c in enumerate(context_chunks)])
        return f"""You are an ultra-fast, highly accurate voice assistant specializing in Goa heritage, geography, culture, and travel.
Context:
{formatted_context}

Question: {query}
Answer concisely, factually, and grounded purely in the provided context."""
