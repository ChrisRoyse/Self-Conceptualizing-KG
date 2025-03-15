"""
Combine all project files into a single text file with proper formatting.
"""

import os
from pathlib import Path

def combine_files():
    # Get current directory
    current_dir = Path.cwd()
    
    # List of files to combine
    files = [
        (current_dir / "selforg/core/manager.py", "Manager Module"),
        (current_dir / "selforg/db/neo4j_client.py", "Neo4j Client Module"),
        (current_dir / "selforg/models/bio_inspired.py", "Bio-Inspired Module"),
        (current_dir / "selforg/models/concept_evolution.py", "Concept Evolution Module"),
        (current_dir / "selforg/models/concept_extractor.py", "Concept Extractor Module"),
        (current_dir / "selforg/models/self_explorer.py", "Self Explorer Module"),
        (current_dir / "selforg/utils/config.py", "Configuration Module"),
        (current_dir / "selforg/__init__.py", "Package Init Module"),
        (current_dir / "run_kg.py", "Main Script"),
        (current_dir / "config.json", "Configuration File"),
        (current_dir / "requirements.txt", "Requirements File")
    ]
    
    # Create output file
    output_path = current_dir / "combined_files.txt"
    print(f"Creating combined file at: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as outfile:
        for filepath, description in files:
            print(f"Processing: {filepath}")
            
            # Write file header
            outfile.write("="*80 + "\n")
            outfile.write(f"File: {filepath.relative_to(current_dir)}\n")
            outfile.write(f"Description: {description}\n")
            outfile.write("="*80 + "\n\n")
            
            try:
                # Check if file exists
                if not filepath.exists():
                    outfile.write(f"ERROR: File does not exist: {filepath}\n\n")
                    print(f"Warning: File not found - {filepath}")
                    continue
                
                # Read and write file contents
                content = filepath.read_text(encoding="utf-8")
                outfile.write(content)
                
                # Add spacing between files
                outfile.write("\n\n" + "-"*80 + "\n\n")
                
            except Exception as e:
                error_msg = f"Error reading file {filepath}: {str(e)}\n\n"
                outfile.write(error_msg)
                print(error_msg)
    
    print(f"\nFiles have been combined into {output_path}")

if __name__ == "__main__":
    combine_files() 