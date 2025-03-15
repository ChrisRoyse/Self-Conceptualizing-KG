"""
Configuration module for managing settings and environment variables.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Configuration manager for the application"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration from file and environment variables"""
        self.config = {
            "neo4j": {
                "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "user": os.getenv("NEO4J_USER", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "NewPassword123"),
                "database": os.getenv("NEO4J_DATABASE", "selforg")
            },
            "model": {
                "path": os.getenv("MODEL_PATH", "granite_3.2_8b_model"),
                "load_in_8bit": False,
                "device_map": "cpu",
                "offload_state_dict": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": "float32",
                "max_memory": None
            },
            "inference": {
                "max_batch_size": 1,
                "max_length": 128,
                "use_cache": True,
                "cpu_offload": True
            },
            "gds": {
                "similarity_threshold": float(os.getenv("GDS_SIMILARITY_THRESHOLD", "0.6")),
                "community_detection": {
                    "algorithm": os.getenv("GDS_COMMUNITY_ALGORITHM", "louvain"),
                    "min_community_size": int(os.getenv("GDS_MIN_COMMUNITY_SIZE", "3"))
                },
                "embedding": {
                    "dimension": int(os.getenv("GDS_EMBEDDING_DIMENSION", "128")),
                    "iterations": int(os.getenv("GDS_EMBEDDING_ITERATIONS", "10"))
                }
            }
        }
        
        # Load from config file if provided
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """Load configuration from a JSON file"""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                # Update config with file values
                for section, values in file_config.items():
                    if section in self.config:
                        if isinstance(self.config[section], dict) and isinstance(values, dict):
                            self._deep_update(self.config[section], values)
                        else:
                            self.config[section] = values
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """Recursively update a dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
    
    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """Get a configuration value"""
        try:
            if key is None:
                return self.config[section]
            return self.config[section][key]
        except KeyError:
            return default
    
    def get_neo4j_config(self) -> Dict[str, str]:
        """Get Neo4j connection configuration"""
        return self.config["neo4j"]
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config["model"]
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration"""
        return self.config["inference"]
    
    def get_gds_config(self) -> Dict[str, Any]:
        """Get Graph Data Science configuration"""
        return self.config["gds"] 