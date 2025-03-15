#!/usr/bin/env python3
"""
Main script for running the self-organizing knowledge graph.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time

from selforg import ConceptNetworkManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("knowledge_graph.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main interactive loop"""
    parser = argparse.ArgumentParser(description='Self-Organizing Knowledge Graph')
    parser.add_argument('--config', type=str, default="config.json",
                      help='Path to configuration file')
    parser.add_argument('--max-concepts', type=int, default=1000,
                      help='Maximum number of concepts to explore')
    parser.add_argument('--no-auto-explore', action='store_true',
                      help='Disable automatic self-exploration')
    args = parser.parse_args()
    
    try:
        # Initialize manager with configuration
        manager = ConceptNetworkManager(config_file=args.config)
        logger.info("Successfully initialized ConceptNetworkManager")
        
        # Start automatic self-exploration unless disabled
        if not args.no_auto_explore:
            manager.start_self_exploration(max_concepts=args.max_concepts)
            logger.info(f"Started automatic self-exploration (max concepts: {args.max_concepts})")
        
    except Exception as e:
        logger.error(f"Error initializing ConceptNetworkManager: {e}")
        print("\nPlease check:")
        print("1. Neo4j server is running and accessible")
        print("2. Configuration file exists and is valid")
        print("3. Model path is correct")
        return 1
    
    print("\n=== Self-Organizing Knowledge Graph ===")
    print("Interactive session started. Type 'help' for commands, 'exit' to quit.\n")
    
    while True:
        try:
            command = input("Enter command: ").strip()
            
            if command == "exit":
                print("\nStopping self-exploration...")
                manager.stop_self_exploration()
                print("Closing connections...")
                manager.close()
                print("Goodbye!")
                break
                
            elif command == "help":
                print("\nAvailable commands:")
                print("  status - Show exploration status")
                print("  start - Start self-exploration")
                print("  stop - Stop self-exploration")
                print("  explore <concept> - Explore a specific concept")
                print("  exit - Close connections and exit")
                
            elif command == "status":
                # Get exploration status
                result = manager.db.execute_query(
                    "MATCH (c:Concept) RETURN count(c) as count"
                )
                concept_count = result[0]['count'] if result else 0
                print(f"\nConcepts in database: {concept_count}")
                print(f"Self-exploration running: {manager.exploration_thread.is_alive() if manager.exploration_thread else False}")
                
            elif command == "start":
                manager.start_self_exploration(max_concepts=args.max_concepts)
                print("Started self-exploration process")
                
            elif command == "stop":
                manager.stop_self_exploration()
                print("Stopped self-exploration process")
                
            elif command.startswith("explore "):
                concept = command[8:].strip()
                result = manager.get_concept_neighborhood(concept)
                if "error" in result:
                    print(f"\n{result['error']}")
                else:
                    print(f"\nConcept: {result['concept']['name']}")
                    print("\nNeighbors:")
                    for n in result.get('neighbors', []):
                        print(f"- {n['neighbor']['name']} (similarity: {n['similarity']:.2f})")
            
            else:
                print("\nUnknown command. Type 'help' for available commands.")
            
        except KeyboardInterrupt:
            print("\nStopping...")
            manager.stop_self_exploration()
            manager.close()
            break
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            print(f"Error: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())