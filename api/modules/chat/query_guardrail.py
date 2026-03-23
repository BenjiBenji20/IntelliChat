import re as regex
from resources.greetings_loader import GREETINGS_SET

_REPEAT_PATTERN = regex.compile(r'([^\s])\1{2,}')

class QueryGuardrail:
    """Normalize and guardrail. Preserves user's original query"""
    def _remove_special_chars(self, query: str) -> str:
        return query.lower().strip().rstrip('?!.,;:_/')

    def _collapse_repeat(self, text: str) -> tuple[str, str]:
        """Returns both collapse-to-1 and collapse-to-2 variants"""
        to_one = _REPEAT_PATTERN.sub(r'\1', text)
        to_two = _REPEAT_PATTERN.sub(r'\1\1', text)
        return to_one, to_two

    def is_greeting(self, query: str) -> bool:
        normalized_query = self._remove_special_chars(query)
        
        if len(normalized_query) > 40:
            return False

        if normalized_query in GREETINGS_SET:
            return True

        collapsed_to_one, collapsed_to_two = self._collapse_repeat(normalized_query)
        
        # check both variants
        return collapsed_to_one in GREETINGS_SET or collapsed_to_two in GREETINGS_SET

query_guardrail = QueryGuardrail()
