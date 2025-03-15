"""
Dynamic concept evolution and adaptation module.

This module implements mechanisms for concepts to:
1. Evolve and adapt based on usage patterns
2. Merge or split based on semantic similarity
3. Form hierarchical relationships dynamically
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from sklearn.cluster import DBSCAN
from sentence_transformers import util

logger = logging.getLogger(__name__)

@dataclass
class ConceptEvolution:
    """Track the evolution of a concept over time"""
    id: str
    name: str
    created_at: datetime
    parent_concepts: List[str]
    child_concepts: List[str]
    embedding: np.ndarray
    usage_count: int
    last_modified: datetime
    merge_history: List[Dict[str, Any]]

class DynamicConceptEvolver:
    """Manages the evolution and adaptation of concepts"""
    
    def __init__(self, 
                 merge_threshold: float = 0.85,
                 split_threshold: float = 0.4,
                 min_usage_for_split: int = 10):
        """Initialize concept evolver"""
        self.concepts: Dict[str, ConceptEvolution] = {}
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        self.min_usage_for_split = min_usage_for_split
    
    def add_concept(self, 
                   concept_id: str,
                   name: str,
                   embedding: np.ndarray,
                   parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a new concept to the evolution system"""
        now = datetime.now()
        
        concept = ConceptEvolution(
            id=concept_id,
            name=name,
            created_at=now,
            parent_concepts=[parent_id] if parent_id else [],
            child_concepts=[],
            embedding=embedding,
            usage_count=1,
            last_modified=now,
            merge_history=[]
        )
        
        self.concepts[concept_id] = concept
        
        # Update parent-child relationships
        if parent_id and parent_id in self.concepts:
            self.concepts[parent_id].child_concepts.append(concept_id)
        
        return self.get_concept_state(concept_id)
    
    def find_merge_candidates(self, concept_id: str) -> List[Dict[str, Any]]:
        """Find concepts that could potentially be merged"""
        if concept_id not in self.concepts:
            return []
            
        concept = self.concepts[concept_id]
        candidates = []
        
        for other_id, other in self.concepts.items():
            if other_id != concept_id:
                similarity = util.cos_sim(
                    concept.embedding.reshape(1, -1),
                    other.embedding.reshape(1, -1)
                )[0][0]
                
                if similarity >= self.merge_threshold:
                    candidates.append({
                        'id': other_id,
                        'name': other.name,
                        'similarity': float(similarity),
                        'usage_count': other.usage_count
                    })
        
        return sorted(candidates, key=lambda x: x['similarity'], reverse=True)
    
    def merge_concepts(self, concept_ids: List[str]) -> Optional[Dict[str, Any]]:
        """Merge multiple concepts into a new concept"""
        if len(concept_ids) < 2:
            return None
            
        # Verify all concepts exist
        concepts = []
        for cid in concept_ids:
            if cid in self.concepts:
                concepts.append(self.concepts[cid])
            else:
                logger.warning(f"Concept {cid} not found for merging")
                return None
        
        now = datetime.now()
        
        # Create merged concept
        merged_embedding = np.mean([c.embedding for c in concepts], axis=0)
        merged_name = "_".join(sorted(c.name for c in concepts))
        merged_id = f"merged_{merged_name}_{now.timestamp():.0f}"
        
        # Record merge history
        merge_record = {
            'timestamp': now.isoformat(),
            'source_concepts': concept_ids,
            'similarity_matrix': [[
                float(util.cos_sim(c1.embedding.reshape(1, -1), 
                                 c2.embedding.reshape(1, -1))[0][0])
                for c2 in concepts
            ] for c1 in concepts]
        }
        
        # Create new merged concept
        merged = ConceptEvolution(
            id=merged_id,
            name=merged_name,
            created_at=now,
            parent_concepts=[],
            child_concepts=concept_ids,
            embedding=merged_embedding,
            usage_count=sum(c.usage_count for c in concepts),
            last_modified=now,
            merge_history=[merge_record]
        )
        
        # Update relationships
        self.concepts[merged_id] = merged
        for concept in concepts:
            concept.parent_concepts.append(merged_id)
        
        return self.get_concept_state(merged_id)
    
    def check_split_candidate(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Check if a concept should be split based on usage patterns"""
        if concept_id not in self.concepts:
            return None
            
        concept = self.concepts[concept_id]
        
        # Only consider splitting if concept has been used enough
        if concept.usage_count < self.min_usage_for_split:
            return None
        
        # Check semantic coherence of child concepts
        if len(concept.child_concepts) >= 2:
            child_embeddings = np.array([
                self.concepts[cid].embedding 
                for cid in concept.child_concepts 
                if cid in self.concepts
            ])
            
            # Use DBSCAN to detect potential clusters
            clustering = DBSCAN(
                eps=1-self.split_threshold,
                min_samples=2,
                metric='cosine'
            ).fit(child_embeddings)
            
            if len(set(clustering.labels_)) > 1:  # Multiple clusters found
                return {
                    'concept_id': concept_id,
                    'num_clusters': len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0),
                    'child_distribution': {
                        label: sum(1 for l in clustering.labels_ if l == label)
                        for label in set(clustering.labels_) if label != -1
                    }
                }
        
        return None
    
    def get_concept_state(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a concept"""
        if concept_id not in self.concepts:
            return None
            
        concept = self.concepts[concept_id]
        return {
            'id': concept.id,
            'name': concept.name,
            'created_at': concept.created_at.isoformat(),
            'parent_concepts': concept.parent_concepts,
            'child_concepts': concept.child_concepts,
            'usage_count': concept.usage_count,
            'last_modified': concept.last_modified.isoformat(),
            'merge_history': concept.merge_history
        } 