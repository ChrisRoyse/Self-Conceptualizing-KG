"""
Core manager module for orchestrating the concept network.
"""

import logging
import time
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import networkx as nx
import os
from queue import Queue
from threading import Thread, Event
import asyncio

from ..db.neo4j_client import Neo4jClient
from ..models.concept_extractor import ConceptExtractor
from ..models.bio_inspired import BioInspiredNetwork
from ..models.concept_evolution import DynamicConceptEvolver
from ..models.self_explorer import SelfExplorer
from ..utils.config import Config

logger = logging.getLogger(__name__)

class ConceptNetworkManager:
    """Manages the self-organizing concept network"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the concept network manager"""
        # Load configuration
        self.config = Config(config_file)
        
        # Initialize Neo4j client
        neo4j_config = self.config.get_neo4j_config()
        self.db = Neo4jClient(**neo4j_config)
        
        # Force CPU settings before any torch operations
        logger.info("Configuring for CPU-only operation...")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
        torch.set_default_device('cpu')
        
        # Load model and tokenizer
        model_config = self.config.get_model_config()
        logger.info("Loading model with CPU optimizations...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["path"],
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None,  # Force CPU
            offload_folder="model_offload"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_config["path"])
        
        # Initialize components
        self.concept_extractor = ConceptExtractor(self.model, self.tokenizer)
        self.bio_network = BioInspiredNetwork()
        self.concept_evolver = DynamicConceptEvolver()
        self.self_explorer = SelfExplorer(self.model, self.tokenizer)
        
        # Initialize in-memory graph
        self.graph = nx.DiGraph()
        
        # Initialize exploration queue and control
        self.exploration_queue = Queue()
        self.stop_exploration = Event()
        self.exploration_thread = None
        
        # Initialize GDS projections only if we have concepts
        self._init_gds_projections(check_concepts=True)
        
        logger.info("ConceptNetworkManager initialized successfully")
    
    def _init_gds_projections(self, check_concepts: bool = False):
        """Initialize Graph Data Science projections"""
        try:
            # Drop existing projections
            try:
                self.db.execute_gds_query("CALL gds.graph.drop('concept_graph', false)")
            except Exception as e:
                if "not found" not in str(e).lower():
                    logger.warning(f"Error dropping existing projection: {e}")
            
            if check_concepts:
                # Check if we have any concepts before creating projection
                result = self.db.execute_query("MATCH (c:Concept) RETURN count(c) as count")
                if not result or result[0]['count'] == 0:
                    logger.info("No concepts found in database, skipping GDS projection")
                    return
            
            # Create new projection
            self.db.execute_gds_query("""
                CALL gds.graph.project(
                    'concept_graph',
                    'Concept',
                    'RELATED_TO',
                    {
                        nodeProperties: ['vector'],
                        relationshipProperties: ['relatedness']
                    }
                )
            """)
            logger.info("GDS projection created successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize GDS projection: {e}")
            logger.info("GDS projections will be created when concepts are added")
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text to extract and store concepts"""
        # Extract concepts
        concepts = self.concept_extractor.extract_concepts_from_text(text)
        
        # Track new additions
        new_concepts = []
        new_relationships = []
        
        # Process each concept through bio-inspired mechanisms
        for concept in concepts:
            # Generate embedding
            embedding = self.concept_extractor.generate_embedding(concept['name'])
            
            # Add to evolution system
            evolved_state = self.concept_evolver.add_concept(
                concept_id=concept['id'],
                name=concept['name'],
                embedding=embedding
            )
            
            # Store in Neo4j with evolved state
            self.db.execute_query(
                """
                MERGE (c:Concept {id: $id})
                SET c.name = $name,
                    c.confidence = $confidence,
                    c.vector = $vector,
                    c.sources = $sources,
                    c.usage_count = $usage_count,
                    c.created_at = $created_at,
                    c.last_modified = $last_modified
                RETURN c
                """,
                {
                    "id": concept['id'],
                    "name": concept['name'],
                    "confidence": concept['confidence'],
                    "vector": embedding.tolist(),
                    "sources": concept.get('sources', ['text']),
                    "usage_count": evolved_state['usage_count'],
                    "created_at": evolved_state['created_at'],
                    "last_modified": evolved_state['last_modified']
                }
            )
            
            # Add to bio-inspired network
            self.bio_network.immune_response(
                concept['id'],
                [{'id': c['id'], 'name': c['name'], 'similarity': 
                  self.concept_extractor.compute_relatedness(concept['name'], c['name'])}
                 for c in concepts if c['id'] != concept['id']]
            )
            
            # Add to in-memory graph
            self.graph.add_node(
                concept['id'],
                name=concept['name'],
                confidence=concept['confidence']
            )
            new_concepts.append(concept)
        
        # Discover relationships using GDS
        self._discover_relationships_gds(concepts, new_relationships)
        
        # Apply homeostatic regulation
        regulations = self.bio_network.homeostatic_regulation()
        
        # Check for potential concept merges using GDS
        self._check_concept_merges_gds(concepts)
        
        return {
            "new_concepts": new_concepts,
            "new_relationships": new_relationships,
            "regulations": regulations
        }
    
    def _discover_relationships_gds(self, concepts: List[Dict[str, Any]], 
                                  new_relationships: List[Dict[str, Any]]):
        """Discover relationships between concepts using Graph Data Science"""
        # Update GDS projection
        self._init_gds_projections()
        
        # Run node similarity
        gds_config = self.config.get("gds", {})
        similarity_threshold = gds_config.get("similarity_threshold", 0.6)
        
        result = self.db.execute_gds_query("""
            CALL gds.nodeSimilarity.stream('concept_graph')
            YIELD node1, node2, similarity
            WHERE similarity >= $threshold
            RETURN gds.util.asNode(node1) AS source,
                   gds.util.asNode(node2) AS target,
                   similarity
        """, {"threshold": similarity_threshold})
        
        for record in result:
            source = record['source']
            target = record['target']
            similarity = record['similarity']
            
            # Add to Neo4j
            self.db.execute_query(
                """
                MATCH (c1:Concept {id: $source_id})
                MATCH (c2:Concept {id: $target_id})
                MERGE (c1)-[r:RELATED_TO]->(c2)
                SET r.relatedness = $relatedness,
                    r.confidence = $confidence,
                    r.last_modified = datetime()
                RETURN r
                """,
                {
                    "source_id": source['id'],
                    "target_id": target['id'],
                    "relatedness": similarity,
                    "confidence": min(similarity + 0.1, 0.95)
                }
            )
            
            # Update bio-inspired network
            self.bio_network.regulate_activation(
                source['id'],
                [target['id']]
            )
            
            # Add to in-memory graph
            self.graph.add_edge(
                source['id'],
                target['id'],
                weight=similarity
            )
            
            new_relationships.append({
                "source": source['id'],
                "target": target['id'],
                "relatedness": similarity
            })
    
    def _check_concept_merges_gds(self, concepts: List[Dict[str, Any]]):
        """Check for potential concept merges using Graph Data Science"""
        gds_config = self.config.get("gds", {})
        community_config = gds_config.get("community_detection", {})
        
        # Run community detection
        result = self.db.execute_gds_query("""
            CALL gds.louvain.stream('concept_graph')
            YIELD nodeId, communityId
            WITH gds.util.asNode(nodeId) AS node, communityId
            WITH communityId, collect(node) as nodes
            WHERE size(nodes) >= $minSize
            RETURN communityId, nodes
        """, {"minSize": community_config.get("min_community_size", 3)})
        
        for record in result:
            community_nodes = record['nodes']
            if len(community_nodes) > 1:
                # Check if nodes should be merged
                node_ids = [node['id'] for node in community_nodes]
                for node_id in node_ids:
                    merge_candidates = self.concept_evolver.find_merge_candidates(node_id)
                    if merge_candidates:
                        logger.info(f"Found merge candidates in community {record['communityId']}: {merge_candidates}")
    
    def get_concept_neighborhood(self, concept_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get the neighborhood of a concept using Graph Data Science"""
        # Find concept by name
        result = self.db.execute_query(
            "MATCH (c:Concept) WHERE c.name = $name RETURN c",
            {"name": concept_name}
        )
        
        if not result:
            return {"error": f"Concept '{concept_name}' not found"}
        
        concept = result[0]['c']
        
        # Get neighborhood using GDS
        neighbors = self.db.execute_gds_query("""
            CALL gds.fastRP.stream('concept_graph', {
                embeddingDimension: $dimension,
                iterationWeights: [1.0],
                relationshipWeightProperty: 'relatedness'
            })
            YIELD nodeId, embedding
            WITH gds.util.asNode(nodeId) AS node, embedding
            WHERE node.id = $conceptId
            CALL gds.knn.stream('concept_graph', {
                topK: 10,
                nodeProperties: ['vector'],
                similarityMetric: 'cosine'
            })
            YIELD node1, node2, similarity
            WHERE gds.util.asNode(node1).id = $conceptId
            RETURN gds.util.asNode(node2) AS neighbor,
                   similarity
        """, {
            "conceptId": concept['id'],
            "dimension": self.config.get("gds", {}).get("embedding", {}).get("dimension", 128)
        })
        
        # Get bio-inspired state
        bio_state = self.bio_network.get_concept_state(concept['id'])
        
        # Get evolution state
        evo_state = self.concept_evolver.get_concept_state(concept['id'])
        
        return {
            "concept": concept,
            "neighbors": neighbors,
            "bio_state": bio_state,
            "evolution_state": evo_state
        }
    
    def close(self):
        """Clean up resources"""
        try:
            self.db.execute_gds_query("CALL gds.graph.drop('concept_graph', false)")
        except:
            pass
        self.concept_extractor.remove_hooks()
        self.db.close()
        logger.info("ConceptNetworkManager closed successfully")
    
    def start_self_exploration(self, max_concepts: int = 1000):
        """Start autonomous exploration of knowledge"""
        if self.exploration_thread and self.exploration_thread.is_alive():
            logger.info("Self-exploration already running")
            return
            
        self.stop_exploration.clear()
        self.exploration_thread = Thread(
            target=self._run_exploration_loop,
            args=(max_concepts,),
            daemon=True
        )
        self.exploration_thread.start()
        logger.info("Started self-exploration process")
        
    def stop_self_exploration(self):
        """Stop the autonomous exploration process"""
        if self.exploration_thread and self.exploration_thread.is_alive():
            self.stop_exploration.set()
            self.exploration_thread.join()
            logger.info("Stopped self-exploration process")
            
    def _run_exploration_loop(self, max_concepts: int):
        """Main exploration loop"""
        try:
            # Generate seed concepts if database is empty
            if not self._has_concepts():
                logger.info("Generating seed concepts...")
                seed_concepts = self.self_explorer.generate_seed_concepts()
                for concept in seed_concepts:
                    self.exploration_queue.put(concept['name'])
                    self._store_concept(concept)
            
            concepts_explored = 0
            while not self.stop_exploration.is_set() and concepts_explored < max_concepts:
                # Get next concept to explore
                try:
                    concept_name = self.exploration_queue.get(timeout=5)
                except Queue.Empty:
                    continue
                
                # Explore the concept
                logger.info(f"Exploring concept: {concept_name}")
                exploration_result = self.self_explorer.explore_concept(concept_name)
                
                # Store new concepts and relationships
                for concept in exploration_result.get('related_concepts', []):
                    if not self._concept_exists(concept['name']):
                        self._store_concept(concept)
                        self.exploration_queue.put(concept['name'])
                
                for rel in exploration_result.get('relationships', []):
                    self._store_relationship(rel)
                
                concepts_explored += 1
                if concepts_explored % 10 == 0:
                    logger.info(f"Explored {concepts_explored} concepts")
                    
        except Exception as e:
            logger.error(f"Error in exploration loop: {e}")
            self.stop_exploration.set()
            
    def _has_concepts(self) -> bool:
        """Check if database has any concepts"""
        result = self.db.execute_query("MATCH (c:Concept) RETURN count(c) as count")
        return result and result[0]['count'] > 0
        
    def _concept_exists(self, name: str) -> bool:
        """Check if concept exists in database"""
        result = self.db.execute_query(
            "MATCH (c:Concept {name: $name}) RETURN count(c) as count",
            {"name": name}
        )
        return result and result[0]['count'] > 0
        
    def _store_concept(self, concept: Dict[str, Any]):
        """Store a concept in Neo4j"""
        self.db.execute_query(
            """
            MERGE (c:Concept {name: $name})
            SET c.id = $id,
                c.confidence = $confidence,
                c.created_at = $created_at,
                c.source = $source
            RETURN c
            """,
            concept
        )
        
    def _store_relationship(self, relationship: Dict[str, Any]):
        """Store a relationship in Neo4j"""
        self.db.execute_query(
            """
            MATCH (c1:Concept {name: $source})
            MATCH (c2:Concept {name: $target})
            MERGE (c1)-[r:RELATED_TO]->(c2)
            SET r.type = $type,
                r.confidence = $confidence,
                r.last_modified = datetime()
            RETURN r
            """,
            relationship
        ) 