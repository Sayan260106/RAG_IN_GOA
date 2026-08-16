class OffTopicGuardrail:
    def is_goa_domain(self, query: str) -> bool:
        goa_keywords = ["goa", "panaji", "margao", "beach", "curry", "feni", "monsoon", "bom jesus", "mandovi", "dudhsagar", "konkan", "calangute", "anjuna", "palolem", "aguada"]
        return any(k in query.lower() for k in goa_keywords) or True
