"""
Concept extraction module for analyzing model activations and text.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
import re

logger = logging.getLogger(__name__)

class ConceptExtractor:
    """Extracts concepts from model activations and text"""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                 sentence_model: Optional[SentenceTransformer] = None):
        """Initialize with model and tokenizer"""
        self.model = model
        self.tokenizer = tokenizer
        self.sentence_model = sentence_model or SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize hooks for accessing internal activations
        self.hooks = []
        self.activations = {}
        self.concept_cache = {}
        self._setup_activation_hooks()
    
    def _setup_activation_hooks(self):
        """Set up hooks to capture model's internal activations"""
        def embedding_hook(module, input, output):
            self.activations['embedding'] = output.detach()
            return output
        
        def attention_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[f'attention_{layer_idx}'] = output[0].detach()
                else:
                    self.activations[f'attention_{layer_idx}'] = output.detach()
                return output
            return hook
        
        try:
            # Add hooks to model layers
            if hasattr(self.model, 'get_input_embeddings'):
                self.hooks.append(
                    self.model.get_input_embeddings().register_forward_hook(embedding_hook)
                )
            
            # Add hooks to attention layers
            for i, layer in enumerate(self.model.model.layers):
                if i % 3 == 0:  # Sample every third layer
                    self.hooks.append(
                        layer.self_attn.register_forward_hook(attention_hook(i))
                    )
        except Exception as e:
            logger.warning(f"Error setting up activation hooks: {e}")
    
    def extract_concepts_from_text(self, text: str, max_concepts: int = 20) -> List[Dict[str, Any]]:
        """Extract concepts from text using model analysis"""
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(tokens)
        
        # Extract concepts using different methods
        attention_concepts = self._extract_from_attention(tokens, outputs)
        linguistic_concepts = self._extract_from_linguistics(text)
        activation_concepts = self._extract_from_activations(tokens, outputs)
        
        # Merge concepts and compute confidence
        all_concepts = {}
        for concept in attention_concepts + linguistic_concepts + activation_concepts:
            concept_id = concept['id']
            if concept_id in all_concepts:
                existing = all_concepts[concept_id]
                existing['confidence'] = min(0.95, existing['confidence'] + 0.1)
                existing['sources'] = list(set(existing.get('sources', []) + concept.get('sources', [])))
            else:
                all_concepts[concept_id] = concept
        
        # Sort by confidence and return top concepts
        sorted_concepts = sorted(
            all_concepts.values(), 
            key=lambda x: x['confidence'], 
            reverse=True
        )
        return sorted_concepts[:max_concepts]
    
    def _extract_from_attention(self, tokens, outputs) -> List[Dict[str, Any]]:
        """Extract concepts based on attention patterns"""
        concepts = []
        try:
            attention = outputs.attentions[-1].mean(dim=1)  # Use last layer
            top_k = min(10, attention.size(-1))
            top_indices = torch.topk(attention.mean(dim=1), top_k).indices
            
            for idx in top_indices[0].cpu().numpy():
                token_text = self.tokenizer.decode([tokens[0][idx].item()])
                if len(token_text.strip()) > 2:
                    concepts.append({
                        'id': f"concept_{token_text.replace(' ', '_')}",
                        'name': token_text.strip(),
                        'confidence': float(attention[0, 0, idx].mean()),
                        'sources': ['attention']
                    })
        except Exception as e:
            logger.warning(f"Error in attention extraction: {e}")
        return concepts
    
    def _extract_from_linguistics(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts using linguistic patterns"""
        concepts = []
        patterns = [
            (r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', 'entity', 0.8),
            (r'\b[a-z]+_[a-z]+\b', 'technical', 0.75),
            (r'\b[A-Z][A-Z]+\b', 'acronym', 0.7)
        ]
        
        for pattern, type_, base_confidence in patterns:
            for match in re.finditer(pattern, text):
                concept_text = match.group(0)
                concepts.append({
                    'id': f"concept_{concept_text.replace(' ', '_')}",
                    'name': concept_text,
                    'confidence': base_confidence,
                    'sources': ['linguistic'],
                    'type': type_
                })
        return concepts
    
    def _extract_from_activations(self, tokens, outputs) -> List[Dict[str, Any]]:
        """Extract concepts based on activation patterns"""
        concepts = []
        try:
            hidden_states = outputs.last_hidden_state[0]
            activations = torch.norm(hidden_states, dim=1)
            top_k = min(10, len(activations))
            top_indices = torch.topk(activations, top_k).indices.cpu().numpy()
            
            for idx in top_indices:
                token_text = self.tokenizer.decode([tokens[0][idx].item()])
                if len(token_text.strip()) > 2:
                    concepts.append({
                        'id': f"concept_{token_text.replace(' ', '_')}",
                        'name': token_text.strip(),
                        'confidence': float(activations[idx] / activations.max()),
                        'sources': ['activation']
                    })
        except Exception as e:
            logger.warning(f"Error in activation extraction: {e}")
        return concepts
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using sentence transformer"""
        if text in self.concept_cache:
            return self.concept_cache[text]
        
        embedding = self.sentence_model.encode(text, show_progress_bar=False)
        self.concept_cache[text] = embedding
        return embedding
    
    def compute_relatedness(self, text1: str, text2: str) -> float:
        """Compute semantic relatedness between two texts"""
        emb1 = self.generate_embedding(text1)
        emb2 = self.generate_embedding(text2)
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def remove_hooks(self):
        """Remove activation hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = [] 