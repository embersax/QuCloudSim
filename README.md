# QuCloudSimPy

A Python-based Quantum Cloud Simulator for distributed quantum computing environments.

## Overview

QuCloudSimPy simulates the behavior of quantum jobs running on distributed quantum cloud infrastructures with multiple networked quantum processors (QPUs).

## Features

- Simulate quantum clouds with various network topologies
- Generate and schedule quantum circuit jobs
- Different scheduling strategies and placement algorithms
- Analyze performance metrics of distributed quantum computing

## Core Components

- **Quantum Cloud Model**: Network of QPUs with configurable topologies
- **Job Management**: Generation and tracking of quantum circuit jobs
- **Scheduling System**: Various strategies for job scheduling and resource allocation
- **Simulation Engine**: Discrete Event Simulator (DES) for modeling system behavior

## Basic Usage

```python
from job import job_generator
from cluster import qCloud
from jobScheduler import job_scheduler
from des import DES
import networkx as nx

# Create a quantum cloud with 4 QPUs in a ring topology
cloud = qCloud(4, nx.cycle_graph, None, 5, 20)

# Generate quantum jobs
job_gen = job_generator()
job_queue = job_gen.generate_job(10)

# Create a discrete event simulator and scheduler
des_simulator = DES(cloud=cloud)
scheduler = job_scheduler(job_queue, des_simulator, cloud)

# Schedule and run jobs
scheduler.schedule_fifo()
des_simulator.run()
```

## Features

- **Circuit Types**: QFT, QuGAN, Arithmetic, and custom circuits
- **Network Topologies**: Ring, Star, Mesh, Random, Custom
- **Scheduling Strategies**: FIFO, Greedy, Simulated Annealing, BFS
- **Placement Algorithms**: Genetic Algorithm, Simulated Annealing, Partitioning

## Requirements

- Python 3.6+
- numpy, networkx, pytket, matplotlib, pymetis

## License

MIT License 