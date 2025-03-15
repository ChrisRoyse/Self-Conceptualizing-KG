"""
Bio-inspired mechanisms for dynamic concept network adaptation.

This module implements biological computing metaphors including:
1. Immune System Metaphor: Concepts as antigens/antibodies that adapt and respond
2. Gene Regulation Network: Concept activation patterns that regulate each other
3. Homeostatic Plasticity: Self-regulation of concept importance
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Any
import networkx as nx
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ConceptMemory:
    """Memory structure for concepts with immune system inspired features"""
    id: str
    name: str
    affinity: float  # How strongly this concept matches with others
    lifetime: float  # How long this concept persists
    activation: float  # Current activation level
    last_used: datetime
    connections: Set[str]  # Connected concept IDs

class BioInspiredNetwork:
    """Bio-inspired network adaptation mechanisms"""
    
    def __init__(self, 
                 affinity_threshold: float = 0.6,
                 decay_rate: float = 0.1,
                 homeostatic_target: float = 0.5):
        """Initialize bio-inspired network"""
        self.memories: Dict[str, ConceptMemory] = {}
        self.affinity_threshold = affinity_threshold
        self.decay_rate = decay_rate
        self.homeostatic_target = homeostatic_target
        self.activation_history: Dict[str, List[float]] = {}
    
    def immune_response(self, concept_id: str, similar_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Implement immune system metaphor where concepts adapt based on similarity
        to other concepts, like antibodies responding to antigens
        """
        if concept_id not in self.memories:
            return []
        
        memory = self.memories[concept_id]
        now = datetime.now()
        
        # Increase affinity for similar concepts (clonal selection)
        adapted_concepts = []
        for similar in similar_concepts:
            similarity = similar.get('similarity', 0)
            if similarity > self.affinity_threshold:
                # Create or update memory for similar concept
                if similar['id'] not in self.memories:
                    self.memories[similar['id']] = ConceptMemory(
                        id=similar['id'],
                        name=similar['name'],
                        affinity=similarity,
                        lifetime=1.0,
                        activation=similarity,
                        last_used=now,
                        connections=set([concept_id])
                    )
                else:
                    sim_memory = self.memories[similar['id']]
                    sim_memory.affinity = max(sim_memory.affinity, similarity)
                    sim_memory.activation += similarity
                    sim_memory.connections.add(concept_id)
                    sim_memory.last_used = now
                
                adapted_concepts.append({
                    'id': similar['id'],
                    'name': similar['name'],
                    'affinity': similarity,
                    'is_new': similar['id'] not in memory.connections
                })
                
                memory.connections.add(similar['id'])
        
        # Update memory
        memory.last_used = now
        memory.activation += len(adapted_concepts) * 0.1
        
        return adapted_concepts
    
    def regulate_activation(self, concept_id: str, connected_concepts: List[str]) -> Dict[str, float]:
        """
        Implement gene regulation network metaphor where concepts regulate
        each other's activation levels
        """
        if concept_id not in self.memories:
            return {}
        
        memory = self.memories[concept_id]
        regulation_effects = {}
        
        # Calculate regulatory effects
        for connected_id in connected_concepts:
            if connected_id in self.memories:
                connected = self.memories[connected_id]
                
                # Compute regulatory effect based on:
                # 1. Affinity between concepts
                # 2. Current activation levels
                # 3. Historical interaction patterns
                
                effect = (
                    connected.affinity * 
                    (connected.activation / (1 + abs(memory.activation - self.homeostatic_target))) *
                    np.exp(-self.decay_rate * (datetime.now() - connected.last_used).total_seconds())
                )
                
                regulation_effects[connected_id] = effect
                
                # Update activation levels
                connected.activation = max(0, min(1, connected.activation + effect * 0.1))
                
                # Store activation history
                if connected_id not in self.activation_history:
                    self.activation_history[connected_id] = []
                self.activation_history[connected_id].append(connected.activation)
        
        return regulation_effects
    
    def homeostatic_regulation(self) -> List[Dict[str, Any]]:
        """
        Implement homeostatic plasticity to maintain balanced activation
        across the network
        """
        regulations = []
        
        # Calculate network-wide statistics
        activations = [m.activation for m in self.memories.values()]
        if not activations:
            return regulations
            
        mean_activation = np.mean(activations)
        std_activation = np.std(activations)
        
        # Regulate each concept
        for concept_id, memory in self.memories.items():
            # Calculate how far this concept is from homeostatic target
            deviation = memory.activation - self.homeostatic_target
            
            # Apply homeostatic regulation
            if abs(deviation) > std_activation:
                regulation = -deviation * self.decay_rate
                memory.activation = max(0, min(1, memory.activation + regulation))
                
                regulations.append({
                    'id': concept_id,
                    'name': memory.name,
                    'old_activation': memory.activation - regulation,
                    'new_activation': memory.activation,
                    'regulation': regulation
                })
        
        return regulations
    
    def prune_inactive_concepts(self, max_inactive_time: float = 3600) -> List[str]:
        """Remove concepts that haven't been active recently"""
        now = datetime.now()
        to_remove = []
        
        for concept_id, memory in self.memories.items():
            inactive_time = (now - memory.last_used).total_seconds()
            if inactive_time > max_inactive_time and memory.activation < 0.1:
                to_remove.append(concept_id)
        
        for concept_id in to_remove:
            del self.memories[concept_id]
            if concept_id in self.activation_history:
                del self.activation_history[concept_id]
        
        return to_remove
    
    def get_concept_state(self, concept_id: str) -> Dict[str, Any]:
        """Get current state of a concept"""
        if concept_id not in self.memories:
            return None
            
        memory = self.memories[concept_id]
        return {
            'id': memory.id,
            'name': memory.name,
            'affinity': memory.affinity,
            'activation': memory.activation,
            'lifetime': memory.lifetime,
            'last_used': memory.last_used.isoformat(),
            'num_connections': len(memory.connections),
            'activation_history': self.activation_history.get(concept_id, [])
        } 