# 🎯 Multi-Agent Pursuit-Evasion Planning System

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

This project implements an advanced multi-agent pursuit-evasion planning system using Monte Carlo Tree Search (MCTS) with probabilistic state transitions. The system simulates a complex three-agent scenario where each agent must simultaneously pursue one target while evading another pursuer, all while navigating through a grid-based environment with obstacles.

## 🚀 Key Features

### Advanced Algorithm Implementation
- **Monte Carlo Tree Search (MCTS)**: Sophisticated decision-making algorithm for optimal path planning
- **Probabilistic State Transitions**: Intelligent handling of action outcomes with probability distributions
- **Multi-Agent Interaction**: Complex three-agent dynamics with simultaneous pursuit and evasion
- **Grid-Based Navigation**: Efficient path planning in obstacle-rich environments

### Performance Optimizations
- **LRU Caching**: Optimized computation of frequently used operations
- **Efficient State Space Exploration**: Smart pruning and exploration strategies
- **Optimized Rollout Policies**: Fast and effective simulation-based decision making
- **Memory-Efficient Implementation**: Careful resource management for large-scale simulations

## 💻 Technical Implementation

### Core Technologies
- **Language**: Python 3.x
- **Key Libraries**: 
  - NumPy: Efficient numerical computations
  - SciPy: Advanced scientific computing capabilities

### Core Algorithms
- **Monte Carlo Tree Search (MCTS)**
  - Selection: UCT-based node selection
  - Expansion: Dynamic tree growth
  - Simulation: Fast rollout policies
  - Backpropagation: Value updates
- **Probabilistic State Transitions**
  - Action rotation handling
  - Probability distribution management
- **Heuristic-based Rollout Policies**
  - Distance-based evaluation
  - Collision avoidance
- **Dynamic Path Planning**
  - Grid-based navigation
  - Obstacle avoidance

## 📁 Project Structure
```
├── planners/
│   ├── planner.py    # Main MCTS implementation
│   ├── tom.py        # Tom agent implementation
│   ├── jerry.py      # Jerry agent implementation
│   └── __init__.py
├── main.py           # Main execution script
└── devel.py          # Development and testing utilities
```

## 📊 Performance Metrics
- **Success Rate**: High success rate in complex multi-agent scenarios
- **Computation Time**: Optimized for real-time decision making
- **Memory Usage**: Efficient resource utilization
- **Scalability**: Handles various grid sizes and agent configurations

## 🔮 Future Enhancements
- **Deep Learning Integration**: Neural network-based policy improvement
- **Parallel Processing**: Multi-threaded MCTS iterations
- **Enhanced Visualization**: Real-time simulation visualization
- **Extended Scenarios**: Support for more complex multi-agent interactions
- **Performance Profiling**: Detailed performance analysis tools

## 🛠️ Requirements
- Python 3.x
- NumPy
- SciPy

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/devendrasaim/multi-agent-pursuit-evasion.git

# Navigate to project directory
cd multi-agent-pursuit-evasion

# Install dependencies
pip install numpy scipy
```

## 🚀 Usage
```python
from planners.planner import PlannerAgent
from main import Task

# Initialize and run a task
task = Task(grid_id=5, running_id=0)
result = task.run()
```

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author
Deven Saim

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/devendrasaim/multi-agent-pursuit-evasion/issues).

## ⭐ Show your support
Give a ⭐️ if this project helped you!
