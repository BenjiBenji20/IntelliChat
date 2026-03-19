class StringNormalizer:
    """Normalize strings for prefix"""
    
    def remove_special_chars(self, query: str) -> str:
        """
        Removes punctuation, converts to lowercase
        """
        return query.lower().strip().rstrip('?!.,;:_/')
    
    def normalize_query_cache_key(self, prefix: str, query: str) -> str:
        """
        Normalize query for consistent caching
        Removes punctuation, converts to lowercase
        """
        # Remove trailing punctuation and convert to lowercase
        return f"{prefix}_{self.remove_special_chars(query)}"
    
str_normalizer = StringNormalizer()
