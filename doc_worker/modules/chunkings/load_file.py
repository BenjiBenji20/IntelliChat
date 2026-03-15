import asyncio
from typing import Dict, List
import json


"""
Loads file according to its file type. Different file, different structure, different loading process.
maintains structure for json, jsonl and md files
"""
class FileLoader:
    """
    Loads raw file bytes into plain text content
    based on file type.
    """
    
    async def load(self, file_bytes: bytes, file_type: str) -> str | List[Dict] | None:
        if file_type == "txt":
            return await asyncio.get_running_loop().run_in_executor(None, self._load_txt_file, file_bytes)
        elif file_type == "md":
            return await asyncio.get_running_loop().run_in_executor(None, self._load_md_file, file_bytes)
        elif file_type == "json":
            return await asyncio.get_running_loop().run_in_executor(None, self._load_json_file, file_bytes)
        elif file_type == "jsonl":
            return await asyncio.get_running_loop().run_in_executor(None, self._load_jsonl_file, file_bytes)
        else:
            return None  # unsupported — document status marks as failed

    
    # json_loader — allow both dict and list items
    def _load_json_file(self, file_bytes: bytes) -> List[Dict] | None:
        data = json.loads(file_bytes.decode("utf-8"))
        
        if isinstance(data, list):
            result = [item for item in data if isinstance(item, (dict, list))]
            return result if result else None
        
        if isinstance(data, dict):
            return [data]
        
        return None
    
    
    def _load_jsonl_file(self, file_bytes: bytes) -> List[Dict]:
        lines = file_bytes.decode("utf-8").strip().splitlines()
        
        validated = []
        for line in lines:
            line = line.strip()
            if not line: # skip empty lines
                continue
            
            try:
                loaded = json.loads(line)
                validated.append(loaded)
            except json.JSONDecodeError:
                continue # skip error in line
            
            except KeyError:
                continue # skip error for missing key in line
            
        return validated if validated else None
    

    def _load_md_file(self, file_bytes: bytes) -> str:
        # keep raw markdown — chunker will split by headers
        return file_bytes.decode("utf-8")
    
    
    def _load_txt_file(self, file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8")

file_loader = FileLoader()