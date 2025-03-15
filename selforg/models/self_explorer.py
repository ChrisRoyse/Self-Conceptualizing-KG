"""
Self-exploration module for automatic knowledge extraction from LLM.
"""

import logging
from typing import List, Dict, Any, Set
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class SelfExplorer:
    """Manages autonomous exploration of LLM knowledge"""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.explored_concepts: Set[str] = set()
        self.exploration_prompts = [
            "What are the key concepts related to {concept}?",
            "How would you define {concept} and its relationships?",
            "What are the fundamental aspects of {concept}?",
            "What concepts are closely connected to {concept}?",
            "How does {concept} relate to other important ideas?"
        ]
        
    def generate_seed_concepts(self) -> List[Dict[str, Any]]:
        """Generate initial seed concepts to begin exploration"""
        prompt = """List the most fundamental concepts across different domains of knowledge.
        Focus on core ideas that connect multiple fields."""
        
        concepts = self._extract_concepts_from_response(self._generate_response(prompt))
        return [self._format_concept(c) for c in concepts]
    
    def explore_concept(self, concept: str) -> Dict[str, Any]:
        """Deeply explore a single concept and its relationships"""
        if concept in self.explored_concepts:
            return {}
            
        self.explored_concepts.add(concept)
        related_concepts = []
        relationships = []
        
        # Explore through multiple prompts
        for prompt_template in self.exploration_prompts:
            prompt = prompt_template.format(concept=concept)
            response = self._generate_response(prompt)
            
            # Extract concepts and relationships
            new_concepts = self._extract_concepts_from_response(response)
            for new_concept in new_concepts:
                if new_concept not in self.explored_concepts:
                    related_concepts.append(self._format_concept(new_concept))
                    # Create relationship
                    relationships.append({
                        "source": concept,
                        "target": new_concept,
                        "type": self._infer_relationship_type(concept, new_concept, response),
                        "confidence": self._calculate_confidence(response)
                    })
        
        return {
            "concept": concept,
            "related_concepts": related_concepts,
            "relationships": relationships
        }
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from LLM"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1024,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _extract_concepts_from_response(self, response: str) -> List[str]:
        """Extract concept names from model response"""
        # Use the model to identify key concepts
        prompt = f"Extract the key concepts from this text as a comma-separated list:\n{response}"
        concept_text = self._generate_response(prompt)
        return [c.strip() for c in concept_text.split(",") if c.strip()]
    
    def _format_concept(self, concept_name: str) -> Dict[str, Any]:
        """Format concept with metadata"""
        return {
            "id": str(uuid.uuid4()),
            "name": concept_name,
            "confidence": 0.8,  # Base confidence for self-explored concepts
            "created_at": datetime.now().isoformat(),
            "source": "self_exploration"
        }
    
    def _infer_relationship_type(self, source: str, target: str, context: str) -> str:
        """Infer the type of relationship between concepts"""
        prompt = f"What is the relationship between {source} and {target}? Context: {context}"
        response = self._generate_response(prompt)
        # Simplified relationship types
        if "part of" in response.lower():
            return "PART_OF"
        elif "type of" in response.lower():
            return "IS_A"
        else:
            return "RELATED_TO"
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score for extracted information"""
        # Simple heuristic based on response length and coherence
        words = response.split()
        if len(words) < 10:
            return 0.5
        return min(0.8 + (len(words) / 1000), 0.95)  # Cap at 0.95 