# SelfOrg: Bio-Inspired Self-Organizing Knowledge Graph for LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j 5.x](https://img.shields.io/badge/neo4j-5.x-brightgreen.svg)](https://neo4j.com/)

## Overview

SelfOrg is a groundbreaking framework that enables Large Language Models (LLMs) to autonomously conceptualize and organize their knowledge into dynamic, evolving knowledge graphs. By combining state-of-the-art language models with biologically-inspired mechanisms, SelfOrg creates a self-organizing conceptual network that mimics how living systems adapt, evolve, and self-regulate.

This project aims to address key challenges in LLM deployment:
- **Reducing model size requirements** by externalizing knowledge representation
- **Decreasing inference power consumption** through more efficient knowledge access
- **Increasing inference speed** by optimizing conceptual pathways
- **Improving model adaptability** with dynamic knowledge structures

## Key Features

### 🧠 Autonomous Knowledge Extraction
- Self-directed concept exploration and discovery
- Extraction of concepts and relationships from model activations
- Dynamic concept development based on usage patterns

### 🧬 Bio-Inspired Mechanisms
- **Immune System Metaphor**: Concepts adapt and respond like antibodies to antigens
- **Gene Regulation Network**: Concept activation patterns that regulate each other
- **Homeostatic Plasticity**: Self-regulation of concept importance to maintain balance

### 📊 Neo4j Graph Database Integration
- Persistent storage of concept networks
- Advanced graph algorithms for similarity detection
- Community detection for concept clustering

### 🌱 Continuous Evolution
- Dynamic merging and splitting of concepts based on semantic similarity
- Automated pruning of inactive concepts
- Formation of hierarchical relationships

## Architecture

SelfOrg consists of five core components:

1. **ConceptNetworkManager**: Orchestrates the entire system, managing data flow between components
2. **ConceptExtractor**: Analyzes model activations and text to identify meaningful concepts
3. **BioInspiredNetwork**: Implements biological computing metaphors for network adaptation
4. **DynamicConceptEvolver**: Handles the evolution and adaptation of concepts over time
5. **SelfExplorer**: Autonomously explores and expands the knowledge graph

## Installation

### Prerequisites
- Python 3.8+
- Neo4j 5.x with APOC and Graph Data Science libraries
- At least 16GB RAM recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/selforg.git
cd selforg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up a Neo4j database:
   - Install and start Neo4j (5.26.4 recommended)
   - Create a new database named `selforg`
   - Install APOC and Graph Data Science plugins
   - Set password to match config or update config.json

4. Prepare your model:
   - Download a compatible model (tested with granite_3.2_8b_model)
   - Update the model path in config.json

## Configuration

The system can be configured through `config.json`:

```json
{
    "neo4j": {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "NewPassword123",
        "database": "selforg"
    },
    "model": {
        "path": "granite_3.2_8b_model",
        "load_in_8bit": false,
        "device_map": null
        // Other model settings...
    },
    "gds": {
        "similarity_threshold": 0.6,
        "community_detection": {
            "algorithm": "louvain",
            "min_community_size": 3
        }
        // Other graph algorithm settings...
    }
}
```

## Usage

### Basic Usage

Run the main script:
```bash
python run_kg.py
```

This will:
1. Load the configured model and database connections
2. Start the self-exploration process
3. Provide an interactive command interface

### Interactive Commands

- `status`: Show exploration status and concept count
- `start`: Start self-exploration process
- `stop`: Stop self-exploration process
- `explore <concept>`: Explore a specific concept and its neighborhood
- `exit`: Close connections and exit

### Advanced Options

```bash
python run_kg.py --config custom_config.json --max-concepts 2000 --no-auto-explore
```

- `--config`: Specify a custom configuration file
- `--max-concepts`: Set maximum number of concepts to explore
- `--no-auto-explore`: Disable automatic self-exploration on startup

## How It Works

### Concept Extraction

The `ConceptExtractor` identifies meaningful concepts from text and model activations using three approaches:
1. **Attention Analysis**: Extracting concepts from attention patterns in the model
2. **Linguistic Patterns**: Identifying named entities, technical terms, and acronyms
3. **Activation Patterns**: Detecting concepts from neuron activation strength

### Biologically-Inspired Adaptation

The `BioInspiredNetwork` implements three biological computing metaphors:

#### Immune System
Concepts act like antibodies that recognize and respond to new information. Similar concepts strengthen connections, while dissimilar ones may trigger new concept formation.

```python
def immune_response(self, concept_id: str, similar_concepts: List[Dict[str, Any]]):
    # Increase affinity for similar concepts (clonal selection)
    for similar in similar_concepts:
        if similarity > self.affinity_threshold:
            # Create memory for this concept...
```

#### Gene Regulation
Concepts regulate each other's activation patterns, forming complex feedback loops that maintain memory and relevance.

#### Homeostatic Plasticity
The system self-regulates to maintain balance, preventing concept dominance or extinction.

```python
def homeostatic_regulation(self):
    # Calculate network-wide statistics
    mean_activation = np.mean(activations)
    std_activation = np.std(activations)
    
    # Apply homeostatic regulation
    if abs(deviation) > std_activation:
        regulation = -deviation * self.decay_rate
        memory.activation = max(0, min(1, memory.activation + regulation))
```

### Dynamic Evolution

The `DynamicConceptEvolver` handles concept lifecycle:

1. **Merging**: Similar concepts may merge into higher-level abstractions
2. **Splitting**: Concepts that represent multiple distinct ideas may split
3. **Hierarchical Organization**: Formation of parent-child relationships

### Self-Exploration

The `SelfExplorer` drives autonomous knowledge discovery:

1. Generates seed concepts to begin exploration
2. Explores related concepts through targeted prompts
3. Infers relationships and connection types
4. Adds new concepts to the exploration queue

## Performance Optimization

SelfOrg implements several optimizations for efficient operation:

1. **CPU-only operation mode**: Functions without GPU requirements
2. **Low memory usage**: Configured for minimum RAM consumption
3. **Efficient graph algorithms**: Using Neo4j GDS for scalable graph operations
4. **Throttled exploration**: Controlled concept discovery to prevent resource exhaustion

## Future Directions

- Integration with multi-modal models for visual concept learning
- Distributed concept exploration across multiple models
- Fine-tuning capabilities using the generated knowledge graph
- Edge computing optimizations for embedded systems

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project draws inspiration from immune system computing, neural plasticity research, and gene regulatory networks
- Special thanks to the Neo4j and HuggingFace teams for their excellent libraries
- Built with IBM's Granite 3.2 8B parameter reasoning model