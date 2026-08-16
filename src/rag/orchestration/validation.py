class InputValidator:
    def validate_query(self, query: str) -> bool:
        return bool(query and len(query.strip()) > 0)
