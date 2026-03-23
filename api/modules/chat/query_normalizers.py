class QueryNormalizer:
    """Normalize strings for prefix"""
    
    def remove_special_chars(self, query: str) -> str:
        """
        Removes punctuation, converts to lowercase
        """
        return query.lower().strip().rstrip('?!.,;:_/')
    
    
query_normalizer = QueryNormalizer()
