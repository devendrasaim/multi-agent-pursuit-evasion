# Multi-Agent Pursuit-Evasion Planning with Probabilistic Transitions

## Project Overview
This project implements an advanced multi-agent pursuit-evasion planning system using Monte Carlo Tree Search (MCTS) with probabilistic state transitions. The system simulates a complex three-agent scenario where each agent must simultaneously pursue one target while evading another pursuer, all while navigating through a grid-based environment with obstacles.

## Key Features
- **Monte Carlo Tree Search (MCTS) Implementation**: Advanced decision-making algorithm for optimal path planning
- **Probabilistic State Transitions**: Implements a sophisticated probability distribution system for action outcomes
- **Multi-Agent Interaction**: Handles complex three-agent dynamics with simultaneous pursuit and evasion
- **Grid-Based Navigation**: Efficient path planning in obstacle-rich environments
- **Performance Optimizations**: 
  - LRU caching for frequently used computations
  - Efficient state space exploration
  - Optimized rollout policies

## Technical Implementation
- **Language**: Python 3.x
- **Key Libraries**: NumPy, SciPy
- **Core Algorithms**:
  - Monte Carlo Tree Search (MCTS)
  - Probabilistic State Transitions
  - Heuristic-based Rollout Policies
  - Dynamic Path Planning

## Project Structure
```
├── planners/
│   ├── planner.py    # Main MCTS implementation
│   ├── tom.py        # Tom agent implementation
│   ├── jerry.py      # Jerry agent implementation
│   └── __init__.py
├── main.py           # Main execution script
└── devel.py          # Development and testing utilities
```

## Performance Metrics
- Successfully handles complex multi-agent scenarios
- Efficient state space exploration with MCTS
- Robust probabilistic action handling
- Optimized for real-time decision making

## Future Enhancements
- Integration of deep learning for improved decision making
- Parallel processing for faster MCTS iterations
- Enhanced visualization tools
- Extended multi-agent scenarios

## Requirements
- Python 3.x
- NumPy
- SciPy

## Installation
```bash
pip install numpy scipy
```

## Usage
```python
from planners.planner import PlannerAgent
from main import Task

# Initialize and run a task
task = Task(grid_id=5, running_id=0)
result = task.run()
```

## License
MIT License

## Author
[Your Name]