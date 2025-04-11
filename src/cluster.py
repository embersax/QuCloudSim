"""
Quantum Cloud Cluster Module for QuCloudSimPy.

This module defines the components of a quantum cloud infrastructure, including:
- Quantum Processing Units (QPUs)
- Communication and computational qubits
- Network topologies connecting multiple QPUs
- Algorithms for qubit placement and circuit partitioning

The module enables simulation of distributed quantum computing by modeling
how quantum circuits are mapped across multiple networked quantum processors.
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from collections import Counter, deque
from itertools import combinations
from networkx import isomorphism
import pulp
import copy
from pytket import Circuit, OpType, qasm
from deap import base, creator, tools, algorithms
from pytket.utils import Graph
import multiprocessing

class annealer:
    """
    Simulated annealing algorithm for optimizing qubit placement.
    
    This class implements simulated annealing to find near-optimal mappings
    of circuit qubits to physical qubits across multiple QPUs, minimizing 
    communication overhead.
    """
    def __init__(self, circuit, cloud):
        """
        Initialize the annealer with a circuit and quantum cloud.
        
        Args:
            circuit: The quantum circuit to be placed
            cloud: The quantum cloud infrastructure
        """
        self.circuit = circuit
        self.cloud = cloud

    def annealing(self, initial_placement, iterations, temperature):
        """
        Execute the simulated annealing algorithm to optimize qubit placement.
        
        Args:
            initial_placement: A dictionary mapping QPU IDs to lists of qubits
            iterations: Number of iterations to run the algorithm
            temperature: Initial temperature for annealing
            
        Returns:
            Tuple of (optimized placement, history of overhead values)
        """
        # Track overhead values throughout the annealing process
        overhead_histy = []
        partition = initial_placement
        qubit_partition_dict = {qubit: key for key, value in partition.items() for qubit in value}
        cost = self.calculate_cost(qubit_partition_dict, self.circuit)
        
        for i in range(iterations):
            # Choose one random qubit in random QPU to swap out
            qpu_to_swap = random.choice(list(partition.keys()))
            qubit_to_swap = random.choice(partition[qpu_to_swap])

            # Find destination QPU
            destination_qpu = random.choice(list(set(partition.keys()) - {qpu_to_swap}))
            potential_new_partition = copy.deepcopy(partition)

            # Case 1: If the destination QPU has available qubits, just move the qubit
            if len(potential_new_partition[destination_qpu]) < self.cloud.network.nodes[destination_qpu]['qpu'].available_qubits:
                potential_new_partition[qpu_to_swap].remove(qubit_to_swap)
                potential_new_partition[destination_qpu].append(qubit_to_swap)
                
            # Case 2: If the destination QPU has no available qubits, swap with another qubit
            elif len(potential_new_partition[destination_qpu]) == self.cloud.network.nodes[qpu_to_swap]['qpu'].available_qubits:
                potential_new_partition[qpu_to_swap].remove(qubit_to_swap)
                potential_new_partition[destination_qpu].append(qubit_to_swap)
                qubit_to_swap_back = random.choice(partition[destination_qpu])
                potential_new_partition[destination_qpu].remove(qubit_to_swap_back)
                potential_new_partition[qpu_to_swap].append(qubit_to_swap_back)
            else:
                print('error', i, {key: len(value) for key, value in potential_new_partition.items()})

            # Calculate new cost and decide whether to accept the new placement
            qubit_partition_dict = {qubit: key for key, value in potential_new_partition.items() for qubit in value}
            new_cost = self.calculate_cost(qubit_partition_dict, self.circuit)
            delta_cost = new_cost - cost

            if delta_cost < 0:  # Accept if cost decreases
                partition = potential_new_partition
                cost = new_cost
                temperature = temperature * 0.99
            else:  # Accept with probability based on temperature
                p = np.exp(-delta_cost / temperature)
                random_number = random.random()
                if random_number < p:
                    alpha = delta_cost / temperature
                    partition = potential_new_partition
                    cost = new_cost
                    
            overhead_histy.append(cost)
            
        return partition, overhead_histy

    def calculate_cost(self, partition, circuit):
        """
        Calculate the communication cost of a placement.
        
        Counts the number of two-qubit gates that span different QPUs,
        weighted by the network distance between those QPUs.
        
        Args:
            partition: Dictionary mapping qubits to QPU IDs
            circuit: The quantum circuit
            
        Returns:
            Total communication cost
        """
        cost = 0
        for command in circuit:
            type = command.op.type
            qubits = command.qubits
            # Only consider two-qubit gates (not measurement or reset operations)
            if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset:
                # If the two qubits are on different QPUs, add the distance to the cost
                if partition[qubits[0]] != partition[qubits[1]]:
                    distance = nx.shortest_path_length(self.cloud.network, partition[qubits[0]], partition[qubits[1]])
                    cost += distance

        return cost


class cp_qubit:
    """
    Represents a computational qubit in a QPU.
    
    Computational qubits are used for quantum computation and can be
    allocated to specific jobs.
    """
    def __init__(self, id, qpu):
        """
        Initialize a computational qubit.
        
        Args:
            id: Unique identifier for this qubit
            qpu: The QPU this qubit belongs to
        """
        self.qid = id
        self.qpu = qpu
        self.occupied = False
        # Job ID that currently occupies this qubit (if any)
        self.job_id = None

    def allocate(self, job_id):
        """
        Allocate this qubit to a specific job.
        
        Args:
            job_id: The ID of the job to allocate this qubit to
        """
        self.occupied = True
        self.job_id = job_id


class cm_qubit:
    """
    Represents a communication qubit in a QPU.
    
    Communication qubits are used for inter-QPU communication and entanglement
    distribution in the quantum network.
    """
    def __init__(self, id, qpu):
        """
        Initialize a communication qubit.
        
        Args:
            id: Unique identifier for this qubit
            qpu: The QPU this qubit belongs to
        """
        self.qid = id
        self.qpu = qpu
        self.occupied = False
        # Job ID that currently occupies this qubit (if any)
        self.job_id = None

    def allocate(self, job_id):
        """
        Allocate this qubit to a specific job.
        
        Args:
            job_id: The ID of the job to allocate this qubit to
        """
        self.occupied = True
        self.job_id = job_id


class switch:
    """
    Represents a network switch connecting multiple QPUs.
    
    Switches enable network communication between different QPUs in the
    quantum cloud architecture.
    """
    def __init__(self, name, qpus, ncm_qubits):
        """
        Initialize a network switch.
        
        Args:
            name: Identifier for the switch
            qpus: List of QPU IDs connected to this switch
            ncm_qubits: Number of communication qubits in the switch
        """
        self.name = name
        self.qpus = qpus
        self.ncm_qubits = ncm_qubits


class qpu:
    """
    Represents a Quantum Processing Unit (QPU) in the quantum cloud.
    
    A QPU contains both computational qubits (for quantum operations) and
    communication qubits (for networking with other QPUs).
    """
    def __init__(self, id, ncm_qubits, ncp_qubits):
        """
        Initialize a QPU.
        
        Args:
            id: Unique identifier for this QPU
            ncm_qubits: Number of communication qubits
            ncp_qubits: Number of computational qubits
        """
        self.qpuid = id
        self.occupied = False
        self.job_id = []
        self.job_status = {}
        self.ncm_qubits = ncm_qubits
        self.ncp_qubits = ncp_qubits
        self.cm_qubits = []
        self.cp_qubits = []
        self.init_qpu()
        self.available_qubits = ncp_qubits
        self.collaboration_data = None

    def allocate_qubits(self, job_id, n):
        """
        Allocate n qubits to a job.
        
        Args:
            job_id: The ID of the job
            n: Number of qubits to allocate
        """
        self.occupied = True
        self.job_id.append(job_id)
        self.available_qubits -= n

    def init_qpu(self):
        """
        Initialize the QPU by creating communication and computational qubits.
        """
        # Create communication qubits
        for i in range(self.ncm_qubits):
            self.cm_qubits.append(cm_qubit(i, self))
        # Create computational qubits
        for i in range(self.ncp_qubits):
            self.cp_qubits.append(cp_qubit(i, self))

    def allocate_job(self, job, n_qubits):
        """
        Allocate a job to this QPU, consuming n_qubits resources.
        
        Args:
            job: The job object to allocate
            n_qubits: Number of qubits required by the job
        """
        self.occupied = True
        self.job_id.append(job.id)
        self.job_status[job.id] = 'running'
        self.available_qubits -= n_qubits

    def free_qubits(self, n_qubits, job):
        """
        Free qubits after a job is completed.
        
        Args:
            n_qubits: Number of qubits to free
            job: The job that is releasing the qubits
        """
        self.job_status[job.id] = 'finished'
        self.available_qubits += n_qubits


class qCloud:
    """
    Represents a quantum cloud infrastructure with multiple networked QPUs.
    
    This class is the central component of the quantum cloud simulator, managing
    the network topology, QPU instances, and providing algorithms for circuit
    placement and partitioning across multiple quantum processors.
    """
    def __init__(self, num_qpus, topology_func, topology_args, ncm_qubits=5, ncp_qubits=30, need_switch=False,
                 swicth_number=0, topology=None):
        """
        Initialize a quantum cloud with specified topology and QPU configuration.
        
        Args:
            num_qpus: Number of QPUs in the cloud
            topology_func: Function to generate the network topology (e.g., nx.cycle_graph)
            topology_args: Arguments for the topology function
            ncm_qubits: Number of communication qubits per QPU
            ncp_qubits: Number of computational qubits per QPU
            need_switch: Whether to include network switches in the topology
            swicth_number: Number of switches to create (if need_switch is True)
            topology: Pre-defined topology to use instead of generating one
        """
        # Generate the topology with switches if needed
        if need_switch:
            self.network = topology_func(num_qpus, topology_args)
            self.qpus = []
            
            # Create QPUs and add them to the network
            for node in range(num_qpus):
                qpu_instance = qpu(node, ncm_qubits, ncp_qubits)
                self.network.nodes[node]['type'] = 'qpu'
                self.network.nodes[node]['qpu'] = qpu_instance
                self.qpus.append(qpu_instance)
                self.network.nodes[node]['available_qubits'] = [qubit for qubit in qpu_instance.cp_qubits if
                                                                not qubit.occupied]
            
            # Add switches to the network
            server_per_switch = num_qpus // swicth_number
            for i in range(swicth_number):
                switch_name = i + num_qpus
                self.network.add_node(switch_name)
                qpu_list = []
                
                # Connect QPUs to the switch
                for j in range(server_per_switch * i, server_per_switch * (i + 1)):
                    self.network.add_edge(switch_name, j)
                    qpu_list.append(j)
                    
                switch_instance = switch(switch_name, qpu_list, 20)
                self.network.nodes[switch_name]['type'] = 'qpu'
                self.network.nodes[switch_name]['switch'] = switch_instance
                
            self.qpu_qubit_num = ncp_qubits
            
            # Visualize the network
            colors = ['lightblue' if self.network.nodes[node]['type'] == 'qpu' else 'lightgreen' for node in
                      self.network]
            pos = nx.spring_layout(self.network)
            nx.draw(self.network, pos, with_labels=True, node_color=colors, edge_color='gray', node_size=2000,
                    font_size=15)
            plt.show()

        # Generate the topology without switches
        else:
            # Use provided topology if available
            if topology is not None:
                self.network = topology
                print("Using the given topology")
            else:
                self.network = topology_func(num_qpus, topology_args)
                
            self.qpus = []
            self.collboration_data = None
            
            # Add QPU instances to the topology
            for node in range(num_qpus):
                qpu_instance = qpu(node, ncm_qubits, ncp_qubits)
                self.network.nodes[node]['type'] = 'qpu'
                self.network.nodes[node]['qpu'] = qpu_instance
                self.qpus.append(qpu_instance)
                self.network.nodes[node]['available_qubits'] = [qubit for qubit in qpu_instance.cp_qubits if
                                                                not qubit.occupied]
                                                                
            self.qpu_qubit_num = ncp_qubits
            self.set_collaboration_data()
            self.ncm_qubits = ncm_qubits
            
    def test_legal(self, partition):
        """
        Check if a partition is legal in terms of QPU qubit capacity.
        
        A partition is legal if each QPU has enough available qubits to
        accommodate the qubits assigned to it.
        
        Args:
            partition: Dictionary mapping QPU IDs to lists of assigned qubits
            
        Returns:
            True if the partition is legal, False otherwise
        """
        for qpu_id in partition.keys():
            if len(partition[qpu_id]) > self.network.nodes[qpu_id]['qpu'].available_qubits:
                    return False
            return True
        
    def calculate_cost_qpu(self, partition, circuit):
        """
        Calculate the communication cost of a circuit partitioned across QPUs.
        
        The cost is based on the sum of distances between QPUs for each
        two-qubit gate that spans different QPUs.
        
        Args:
            partition: Dictionary mapping qubits to QPU IDs
            circuit: The quantum circuit
            
        Returns:
            Total communication cost
        """
        cost = 0
        for command in circuit:
            type = command.op.type
            qubits = command.qubits
            if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset:
                part_1 = partition[qubits[0]]
                part_2 = partition[qubits[1]]

                if partition[qubits[0]] != partition[qubits[1]]:
                    distance = nx.shortest_path_length(self.network, partition[qubits[0]], partition[qubits[1]])
                    cost += distance

        return cost      

    def _whether_imrpovable(self, qpu_to_qubit):
        """
        Check if the current QPU-to-qubit assignment can be improved.
        
        An assignment is improvable if there are at least two QPUs that have
        room for more qubits.
        
        Args:
            qpu_to_qubit: Dictionary mapping QPU IDs to lists of qubits
            
        Returns:
            True if the assignment can be improved, False otherwise
        """
        count = 0
        for qpu_id in qpu_to_qubit.keys():
            qubits = self.network.nodes[qpu_id]['qpu'].available_qubits
            if qubits > len(qpu_to_qubit[qpu_id]):
                count += 1
        if count > 1:
            return True
        return False
        
    def find_best_imrpovable_qubit(self, smallest_qpu, qpu_to_qubit, wig): 
        """
        Find the best qubit to move from one QPU to another to improve placement.
        
        This method identifies a qubit that, when moved, would reduce the overall
        communication cost based on the weighted interaction graph (wig).
        
        Args:
            smallest_qpu: QPU ID with the smallest number of assigned qubits
            qpu_to_qubit: Dictionary mapping QPU IDs to lists of qubits
            wig: Weighted interaction graph representing qubit interactions
            
        Returns:
            Tuple of (qubit to move, destination QPU ID)
        """
        result = None
        largest_qpu = None
        max_weight = 0
        
        for qubit in qpu_to_qubit[smallest_qpu]:
            candidates = [qpu for qpu in qpu_to_qubit.keys() 
                         if len(qpu_to_qubit[qpu]) < self.network.nodes[qpu]['qpu'].available_qubits 
                         and qpu != smallest_qpu]
                         
            for candidate in candidates:
                # Calculate the net benefit of moving the qubit
                sum = 0
                # Add positive gain from new connections
                for qubit_x in qpu_to_qubit[candidate]:
                    if wig.has_edge(qubit, qubit_x):
                        sum += wig[qubit][qubit_x]['weight']
                # Subtract loss from broken connections
                for qubit_y in qpu_to_qubit[smallest_qpu]:
                    if qubit_y != qubit and wig.has_edge(qubit, qubit_y):
                        sum -= wig[qubit][qubit_y]['weight']
                        
                if sum > max_weight:
                    max_weight = sum
                    result = qubit
                    largest_qpu = candidate

        return result, largest_qpu
        
    def improve_placement(self, placement):
        """
        Improve an existing placement by moving qubits between QPUs.
        
        This method iteratively moves qubits to reduce the overall communication
        cost based on the qubit interaction graph.
        
        Args:
            placement: Current placement object containing partition and other data
            
        Returns:
            Updated placement object or None if no improvement was found
        """
        wig = placement.wig
        node_to_int = {node: i for i, node in enumerate(wig.nodes())}
        int_to_node = {i: node for i, node in enumerate(wig.nodes())}
        qpu_mapping = placement.qpu_mapping[0]
        
        try:
            reverse_mapping = {value: key for key, value in qpu_mapping.items()}
        except:
            print("Error creating reverse mapping")
            return None
        
        # Create a partition mapping qubits to QPUs
        partition = {node: qpu_mapping[placement.partition[node_to_int[node]]] for node in wig.nodes()}
        old_cost = self.calculate_cost_qpu(partition, placement.job.circuit)
        
        # Create a mapping from QPUs to qubits
        qpu_to_qubit = {qpu: [] for qpu in qpu_mapping.values()}
        for key, value in partition.items():
            qpu_to_qubit[value].append(key)
            
        adjusted_edge = set()
        tested_qpu = set()
        
        # Iteratively improve the placement
        while self._whether_imrpovable(qpu_to_qubit):
            smallest_qpu = min(qpu_to_qubit.keys(), key=lambda x: len(qpu_to_qubit[x]))
            qubit, largest_qpu = self.find_best_imrpovable_qubit(smallest_qpu, qpu_to_qubit, wig)
            
            if qubit is None:
                break
            else:
                # Move the qubit from smallest_qpu to largest_qpu
                qpu_to_qubit[smallest_qpu].remove(qubit)
                qpu_to_qubit[largest_qpu].append(qubit)
                partition[qubit] = largest_qpu
                
        old_partition = placement.partition
        
        # Recompute the communication cost
        new_cost = self.calculate_cost_qpu(partition, placement.job.circuit)
        
        # If new cost is higher, no improvement was found
        if old_cost < new_cost:
            return None
            
        # Save original data for comparison
        placement.old_partition = old_partition
        placement.old_rmote_dag = placement.remote_dag
        placement.old_modified_circuit = placement.modified_circuit

        # Reconstruct the circuit with the new partition
        new_modified_circuit, new_remote_dag = self.reconstruct_circuit(partition, placement)
        placement.modified_circuit = new_modified_circuit
        placement.remote_dag = new_remote_dag  
        placement.qpu_to_dict = qpu_to_qubit     
        
        # Update the partition in the placement object
        for i in range(len(placement.partition)):
            item = placement.partition[i]
            node = int_to_node[i]
            qpu = partition[node]
            part = reverse_mapping[qpu]
            if part != placement.partition[i]:
                placement.partition[i] = part
                
        new_partition = placement.partition
        print("Finished improvement")
        placement.cost = new_cost
        return placement
   
    def reconstruct_circuit(self, partition, placement):
        """
        Reconstruct a circuit based on a new partition.
        
        Creates a new circuit containing only the gates that span different QPUs,
        representing the remote operations that will require communication.
        
        Args:
            partition: Dictionary mapping qubits to QPU IDs
            placement: Placement object containing the original circuit
            
        Returns:
            Tuple of (modified circuit, directed acyclic graph)
        """
        circuit = placement.job.circuit
        modified_circuit = Circuit()
        
        # Add all qubits to the new circuit
        for qubit in circuit.qubits:
            modified_circuit.add_qubit(qubit)
            
        # Only add two-qubit gates that span different QPUs
        for command in circuit:
            op_type = command.op.type
            qubits = list(command.args)
            if len(qubits) == 2 and op_type != OpType.Measure:
                node1 = qubits[0]
                node2 = qubits[1]
                part_1 = partition[node1]
                part_2 = partition[node2]
                if part_1 != part_2:
                    # Add the gate to the modified circuit
                    modified_circuit.add_gate(op_type, qubits)
                    
        # Convert to a directed acyclic graph representation
        g = Graph(modified_circuit)
        dag = g.as_nx()
        return modified_circuit, dag




    def ga_find_placement_multi(self, circuit):
        """
        Find optimal qubit-to-QPU placement using a multi-population genetic algorithm.
        
        This method implements a genetic algorithm that evolves a population of
        placement solutions, searching for an optimal mapping of qubits to QPUs
        that minimizes communication costs.
        
        Args:
            circuit: The quantum circuit to place
            
        Returns:
            Tuple of (best partition, final cost)
        """
        n_qubits = circuit.n_qubits
        qpu_list = list(self.find_connected_qpus(n_qubits))

        if not qpu_list:
            raise ValueError("No connected QPUs found that can accommodate the circuit.")

        # Define fitness function and individual type for genetic algorithm
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMin)

        def init_individual():
            """Create a random initial placement."""
            partition = self.random_find_placement(circuit)
            individual = creator.Individual(partition)
            return individual

        def mutate(individual):
            """
            Mutate an individual by swapping qubits between QPUs.
            
            Randomly selects two QPUs and swaps qubits between them.
            """
            partition = individual
            qpu_list = list(partition.keys())

            # Select two different QPUs
            qpu_to_swap = random.choice(qpu_list)
            destination_qpu = random.choice(qpu_list)
            while qpu_to_swap == destination_qpu:
                destination_qpu = random.choice(qpu_list)

            # Select qubits to swap
            qubit_to_swap = random.choice(partition[qpu_to_swap])
            des_qubit_to_swap = random.choice(partition[destination_qpu])

            # Perform the swap
            partition[qpu_to_swap].remove(qubit_to_swap)
            partition[destination_qpu].remove(des_qubit_to_swap)
            partition[qpu_to_swap].append(des_qubit_to_swap)
            partition[destination_qpu].append(qubit_to_swap)

            return individual,

        def crossover(ind1, ind2):
            """
            Perform crossover between two individuals.
            
            This crossover looks for two-qubit gates that cross QPU boundaries
            and tries to place them on the same QPU in the offspring.
            """
            partition1 = ind1
            partition2 = ind2

            for command in circuit:
                type = command.op.type
                qubits = command.qubits
                swap_prb = random.random()

                # Only consider two-qubit gates with a small probability
                if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset and swap_prb < 0.1:
                    # Find QPUs where qubits are placed in each individual
                    p1_qpu_i = None
                    p1_qpu_j = None
                    for pi in partition1:
                        if qubits[0] in partition1[pi]:
                            p1_qpu_i = pi
                        if qubits[1] in partition1[pi]:
                            p1_qpu_j = pi

                    p2_qpu_i = None
                    p2_qpu_j = None
                    for pi in partition2:
                        if qubits[0] in partition2[pi]:
                            p2_qpu_i = pi
                        if qubits[1] in partition2[pi]:
                            p2_qpu_j = pi

                    # If the qubits are on different QPUs in both individuals,
                    # try to swap their placements
                    if p1_qpu_i != p2_qpu_i and p1_qpu_j != p2_qpu_j:
                        if (len(partition1[p2_qpu_i]) < self.qpu_qubit_num and 
                            len(partition1[p2_qpu_j]) < self.qpu_qubit_num and 
                            len(partition2[p1_qpu_i]) < self.qpu_qubit_num and 
                            len(partition2[p1_qpu_j]) < self.qpu_qubit_num):
                            # Move qubits in partition1
                            partition1[p2_qpu_i].append(qubits[0])
                            partition1[p2_qpu_j].append(qubits[1])
                            partition1[p1_qpu_i].remove(qubits[0])
                            partition1[p1_qpu_j].remove(qubits[1])
                            
                            # Move qubits in partition2
                            partition2[p1_qpu_i].append(qubits[0])
                            partition2[p1_qpu_j].append(qubits[1])
                            partition2[p2_qpu_i].remove(qubits[0])
                            partition2[p2_qpu_j].remove(qubits[1])
                            
            # Verify that the resulting partitions are still legal
            assert self.test_legal(partition1) and self.test_legal(partition2)

            return ind1, ind2

        # Set up DEAP toolbox
        toolbox = base.Toolbox()
        toolbox.register("individual", init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)

        def evaluate(individual):
            """
            Evaluate the fitness of an individual.
            
            Calculates the communication cost of the placement and adds a large
            penalty if the placement is not legal.
            """
            cost = self.calculate_cost(individual, circuit)
            if not self.test_legal(individual):
                return 100000 + cost,
            return cost,

        toolbox.register("evaluate", evaluate)

        # Use multiprocessing for parallel evaluation
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

        # Run the genetic algorithm
        population = toolbox.population(n=50)
        ngen = 10  # Number of generations
        cxpb = 0.7  # Crossover probability
        mutpb = 0.2  # Mutation probability

        result_population, logbook = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

        # Clean up multiprocessing resources
        pool.close()
        pool.join()

        # Get the best solution
        best_individual = tools.selBest(result_population, 1)[0]
        partition = best_individual
        final_cost = self.calculate_cost(partition, circuit)
        
        return partition, final_cost

    def calculate_cost(self, partition, circuit):
        """
        Calculate the communication cost of a placement.
        
        Args:
            partition: Dictionary mapping QPUs to lists of qubits
            circuit: The quantum circuit
            
        Returns:
            Total communication cost
        """
        # Convert the partition format to {qubit: qpu_id}
        partition = {value: key for key, values in partition.items() for value in values}
        
        cost = 0
        for command in circuit:
            type = command.op.type
            qubits = command.qubits
            if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset:
                # If the two qubits are on different QPUs, add the distance to the cost
                if partition[qubits[0]] != partition[qubits[1]]:
                    cost += nx.shortest_path_length(self.network, partition[qubits[0]], partition[qubits[1]])

        return cost

    def connect_qpus(self, qpu1_id, qpu2_id):
        """
        Add a direct connection between two QPUs in the network.
        
        Args:
            qpu1_id: ID of the first QPU
            qpu2_id: ID of the second QPU
        """
        self.network.add_edge(qpu1_id, qpu2_id)

    def get_communities(self):
        """
        Detect communities in the QPU network.
        
        Uses the greedy modularity communities algorithm to find groups of
        densely connected QPUs.
        
        Returns:
            List of communities, where each community is a frozenset of QPU IDs
        """
        # Only consider nodes that are QPUs with available qubits
        server_nodes = [node for node in self.network.nodes() if
                        self.network.nodes[node]['type'] == 'qpu' and 
                        self.network.nodes[node]['qpu'].available_qubits > 0]
        server_graph = self.network.subgraph(server_nodes)
        
        # Run community detection algorithm
        communities = nx.community.greedy_modularity_communities(server_graph, weight='weight')
        return communities

    def get_available_qubits(self):
        """
        Get the total number of available qubits across all QPUs.
        
        Returns:
            Total number of available qubits
        """
        return sum(qpu.available_qubits for qpu in self.qpus)

    def contract_wig(self, wig, partition):
        """
        Contract a weighted interaction graph based on a partition.
        
        Creates a new graph where nodes correspond to parts of the partition
        and edges represent interactions between parts.
        
        Args:
            wig: Weighted interaction graph of qubits
            partition: List where index is qubit ID and value is partition ID
            
        Returns:
            Contracted graph where nodes are partition IDs
        """
        # Create mappings between node indices and original nodes
        node_to_int = {node: i for i, node in enumerate(wig.nodes())}
        int_to_node = {i: node for node, i in node_to_int.items()}
        
        # Count qubits in each partition
        count = Counter(partition)
        
        # Group qubits by partition
        qubits_in_partitions = {i: [] for i in count.keys()}
        for node_index, partition_id in enumerate(partition):
            original_node = int_to_node[node_index]
            try:
                qubits_in_partitions[partition_id].append(original_node)
            except:
                print('No partition')
                
        # Create mapping from qubit to its partition
        qubit_partition_dict = {}
        for key, value in qubits_in_partitions.items():
            for qubit in value:
                qubit_partition_dict[qubit] = key
                
        # Create the contracted graph
        graph = nx.Graph()
        for edge in wig.edges():
            # Only consider edges between different partitions
            if qubit_partition_dict[edge[0]] != qubit_partition_dict[edge[1]]:
                # If the edge doesn't exist yet, create it
                if not graph.has_edge(qubit_partition_dict[edge[0]], qubit_partition_dict[edge[1]]):
                    graph.add_edge(qubit_partition_dict[edge[0]], qubit_partition_dict[edge[1]],weight=wig[edge[0]][edge[1]]['weight'])
                # Otherwise, add to the weight
                else:
                    graph[qubit_partition_dict[edge[0]]][qubit_partition_dict[edge[1]]]['weight'] += \
                        wig[edge[0]][edge[1]]['weight']

        return graph

    def compute_weighted_length(self, partition, wig, result):
        """
        Compute the weighted path length for a mapping of qubits to QPUs.
        
        Args:
            partition: Partition of qubits (used for counting)
            wig: Weighted interaction graph of qubits
            result: Mapping from qubits to QPUs
            
        Returns:
            Tuple of (original sum of weights, weighted sum of path lengths)
        """
        total_weighted_sum = 0
        original_sum = 0
        
        for edge in wig.edges():
            node_1, node_2 = edge
            try:
                qpu_1, qpu_2 = result[node_1], result[node_2]
            except:
                print('No result')
                continue
                
            # Calculate distance between QPUs
            distance = nx.shortest_path_length(self.network, qpu_1, qpu_2)
            
            # Add weighted distance to the total
            total_weighted_sum += wig[edge[0]][edge[1]]['weight'] * distance
            
            # Keep track of original sum of weights
            original_sum += wig[edge[0]][edge[1]]['weight']

        return original_sum, total_weighted_sum

    def ramdom_find_placement(self, circuit):
        """
        Find an initial random placement for a circuit.
        
        Randomly assigns qubits to QPUs, ensuring that each QPU's capacity
        is not exceeded.
        
        Args:
            circuit: The quantum circuit to place
            
        Returns:
            Dictionary mapping QPU IDs to lists of qubits
        """
        # Get a copy of the circuit's qubits
        qubits_list = copy.deepcopy(circuit.qubits)
        nqubits = len(qubits_list)
        
        # Find a set of connected QPUs that can accommodate the circuit
        candidate_qpus = self.find_connected_qpus(nqubits)
        qpu_qubit_usage = {qpu: 0 for qpu in candidate_qpus}
        
        # Randomly assign qubits to QPUs
        qubit_mapping = {}
        random.shuffle(qubits_list)
        for qubit in qubits_list:
            for qpu in candidate_qpus:
                if qpu_qubit_usage[qpu] + 1 <= self.qpu_qubit_num:
                    qubit_mapping[qubit] = qpu
                    qpu_qubit_usage[qpu] += 1
                    break
                    
        # Convert to desired output format
        partition = {qpu: [] for qpu in candidate_qpus}
        for key, value in qubit_mapping.items():
            partition[value].append(key)
            
        return partition

    def sa_find_placement(self, circuit):
        """
        Find optimal placement using simulated annealing.
        
        Uses the annealer class to perform simulated annealing optimization
        of qubit placement.
        
        Args:
            circuit: The quantum circuit to place
            
        Returns:
            Tuple of (optimized partition, final cost)
        """
        # Find connected QPUs and create initial placement
        n_qubits = circuit.n_qubits
        qpu_list = self.find_connected_qpus(n_qubits)
        initial_placement = self.ramdom_find_placement(circuit)
        
        # Create annealer and run the annealing process
        annealing_solver = annealer(circuit, self)
        partition, history = annealing_solver.annealing(initial_placement, 1000, 100)
        
        return partition, history[-1]

    def find_placement_bfs(self, comb, wig):
        """
        Find placement for multiple circuit partitions using breadth-first search.
        
        This method tries to find QPU mappings for different partitioning options
        of a circuit using BFS to explore the QPU network.
        
        Args:
            comb: Dictionary where keys are number of QPUs and values are lists of partitions
            wig: Weighted interaction graph representing the circuit
            
        Returns:
            Dictionary of valid placements for each partition option
        """
        # Check if there are enough qubits available globally
        all_available_qubits = self.get_available_qubits()
        for key, value in comb.items():
            for partition in value:
                qubits_needed = len(partition)
                if qubits_needed > all_available_qubits:
                    return []
                    
        # Find placements for each partition option
        res = {}
        for key, value in comb.items():
            res[key] = []
            for partition in value:
                # Contract the graph based on the partition
                remote_wig = self.contract_wig(wig, partition)
                count = Counter(partition)
                
                # Skip if any partition requires more qubits than a single QPU can provide
                if any(element > self.qpu_qubit_num for element in count.values()):
                    continue
                    
                # Try to find a placement using BFS
                    start_node = random.randint(0, len(self.qpus) - 1)
                result = self.bfs_find_qpus(start_node, partition, remote_wig)
                if result:
                    res[key].append(result[0])
                    
        return res

    def find_placement(self, comb, wig):
        """
        Find placement for multiple circuit partitions using community detection.
        
        This method tries to find QPU mappings for different partitioning options
        of a circuit using community detection to identify densely connected QPU groups.
        
        Args:
            comb: Dictionary where keys are number of QPUs and values are lists of partitions
            wig: Weighted interaction graph representing the circuit
            
        Returns:
            Dictionary of valid placements for each partition option
        """
        # Check if there are enough qubits available globally
        all_available_qubits = self.get_available_qubits()
        for key, value in comb.items():
            for partition in value:
                qubits_needed = len(partition)
                if qubits_needed > all_available_qubits:
                    return []
                    
        # Find placements for each partition option
        res = {}
        for key, value in comb.items():
            res[key] = []
            for partition in value:
                # Contract the graph based on the partition
                remote_wig = self.contract_wig(wig, partition)
                count = Counter(partition)
                
                # Skip if any partition requires more qubits than a single QPU can provide
                if any(element > self.qpu_qubit_num for element in count.values()):
                    continue
                    
                # Try to find a placement using community detection
                    result = self.community_find_qpus_weight(partition, remote_wig)
                    if result:
                        res[key].append(result[0])
                    
        return res

    def test_sub_isomorphic(self, communities, remote_wig):
        """
        Test if a remote interaction graph is subgraph-isomorphic to any community.
        
        Checks if the remote_wig can be embedded in any of the communities in the network.
        
        Args:
            communities: List of communities to check
            remote_wig: Remote weighted interaction graph
            
        Returns:
            Mapping dictionary if isomorphism is found, None otherwise
        """
        for community in communities:
            # Get the subgraph for the community
            sub_graph = self.network.subgraph(community)
            
            # Check for subgraph isomorphism
            matcher = isomorphism.GraphMatcher(sub_graph, remote_wig)
            if matcher.subgraph_is_isomorphic():
                print(matcher.mapping)
                print('Isomorphic')
                return matcher.mapping

            else:
                print('Not isomorphic')
                
        return None

    def map_qpu_to_community(self, community, remote_wig):
        """
        Map circuit partitions (nodes in remote_wig) to QPUs in a community.
        
        Uses a greedy approach based on node centrality and degree.
        
        Args:
            community: Set of QPU IDs representing a community
            remote_wig: Remote weighted interaction graph representing the partitioned circuit
            
        Returns:
            Dictionary mapping remote_wig nodes to QPU IDs
        """
        # Create subgraph for the community
        subgraph = self.network.subgraph(community)
        
        # Sort nodes in remote_wig by their weighted degree (most connected first)
        sorted_nodes = sorted(remote_wig.nodes(), key=lambda x: remote_wig.degree(x, weight='weight'), reverse=True)
        
        result = {}
        for node_1 in sorted_nodes:
            # Find the most suitable node in the community for this remote node
            try:
                # Try to find the center of the remaining subgraph
                center_node = nx.center(subgraph)
                if len(center_node) > 1:
                    # If multiple centers, pick the one with highest degree
                    max_degree = 0
                    for node in center_node:
                        if subgraph.degree(node) > max_degree:
                            max_degree = subgraph.degree(node)
                            center_node = [node]
                    result[node_1] = center_node[0]
                else:
                    result[node_1] = center_node[0]
            except:
                # If can't find center (e.g., disconnected graph), pick highest degree node
                center_node = list(subgraph.nodes())
                max_degree = 0
                for node in center_node:
                    if subgraph.degree(node) > max_degree:
                        max_degree = subgraph.degree(node)
                        center_node = [node]
                result[node_1] = center_node[0]
                
            # Remove the assigned node from consideration for future mappings
            rest_nodes = subgraph.nodes() - {center_node[0]}
            subgraph = nx.subgraph(subgraph, rest_nodes)
            
        return result

    def community_find_qpus_weight(self, partition, remote_wig):
        """
        Find a mapping from partitions to QPUs based on community detection.
        
        This method finds communities in the QPU network and tries to map
        the partitions to these communities to minimize communication cost.
        
        Args:
            partition: Dictionary of partition assignments
            remote_wig: Remote weighted interaction graph
            
        Returns:
            Tuple of (mapping dictionary, weighted path length) or None if no mapping found
        """
        # Visualize the remote interaction graph
        nx.draw(remote_wig, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=15)
        plt.show()
        
        # Create a weight graph and detect communities
        self.create_new_weight_graph()
        communities = self.get_communities()
        
        # Calculate total qubits available in each community
        total_qubits_list = [sum([self.network.nodes[node]['qpu'].available_qubits for node in community]) for community in communities]
                             
        qubit_per_qpu = {qpu: self.network.nodes[qpu]['qpu'].available_qubits for community in communities for qpu in community}
                         
        count = Counter(partition)
        
        # First try to find a single community that can accommodate the entire circuit
        single_community = list(filter(
            lambda x: sum([self.network.nodes[node]['qpu'].available_qubits for node in x]) >= sum(count.values()) 
                     and len(x) >= len(count), 
            communities))
            
        # Try to map to each viable community
        result = list(map(lambda x: self.map_remote_wig_to_community_new(remote_wig, x, partition), single_community))
        
        # Filter out unsuccessful mappings
        result = list(filter(lambda x: x != (None, None) and x is not None, result))
        
        if result:
            # Return the best mapping (with smallest weighted path length)
            try:
                return min(result, key=lambda x: x[1])
            except:
                print('Not possible to find optimal mapping')
    
        # If no single community works, try combinations of communities
        possible_combinations = []
        for r in range(2, len(communities) + 1):
            for combo in combinations(communities, r):
                # Calculate total qubits in this combination of communities
                total_qubits_combo = sum(
                    sum(self.network.nodes[node]['qpu'].available_qubits for node in community) 
                    for community in combo)
                    
                # Skip if not enough qubits
                if sum(count.values()) > total_qubits_combo:
                    continue
                    
                # Calculate total number of QPUs
                total_qpu_numbers = sum(len(community) for community in combo)
                
                # Only consider if we have at least as many QPUs as partitions
                if len(count) <= total_qpu_numbers:
                    possible_combinations.append(combo)
        
        # Merge the communities
        merged_sets = []
        for frozensets in possible_combinations:
            merged_frozenset = frozenset().union(*frozensets)
            merged_sets.append(merged_frozenset)
            
        # Find minimal merged sets (not supersets of others)
        minimal_sets = []
        for current_set in merged_sets:
            if all(not current_set.issuperset(other_set) for other_set in merged_sets if current_set != other_set):
                minimal_sets.append(current_set)
                
        # Only consider connected merged communities
        minimal_sets = list(filter(lambda x: nx.is_connected(self.network.subgraph(x)), minimal_sets))
        
        # Try to map to each merged community
        result = list(map(lambda x: self.map_remote_wig_to_community_new(remote_wig, x, partition), minimal_sets))
        result = list(filter(lambda x: x != (None, None) and x is not None, result))
        
        if result:
            return min(result, key=lambda x: x[1])
            
        # No mapping found
        return None

    def check_availability(self, comb, wig):
        """
        Check if there are available resources for a set of placement combinations,
        and find different placement strategies.
        
        Args:
            comb: Dictionary where keys are number of QPUs and values are lists of partitions
            wig: Weighted interaction graph
            
        Returns:
            Tuple of dictionaries with different placement strategies
        """
        # Check if there are enough qubits available globally
        all_available_qubits = self.get_available_qubits()
        for key, value in comb.items():
            for partition in value:
                qubits_needed = len(partition)
                if qubits_needed > all_available_qubits:
                    return []
                    
        # Initialize result dictionaries for different placement strategies
        res = {}  # BFS strategy
        res_community = {}  # Community detection strategy
        length_community = {}  # Length for community strategy
        res_enumerate = {}  # Enumeration strategy
        length = {}  # Length for BFS strategy
        
        for key, value in comb.items():
            res[key] = []
            res_community[key] = []
            res_enumerate[key] = []
            
            for partition in value:
                # Contract the graph based on the partition
                remote_wig = self.contract_wig(wig, partition)
                count = Counter(partition)
                
                # Skip if any partition requires more qubits than a single QPU can provide
                if any(element > self.qpu_qubit_num for element in count.values()):
                    continue
        else:
                    # Try different placement strategies
                    start_node = random.randint(0, len(self.qpus) - 1)
                    
                    # Community-based placement with weights
                    community_with_weight, length_withweight = self.community_find_qpus_weight(partition, remote_wig)
                    
                    # BFS-based placement
                    available_qpus, weighted_length_bfs = self.bfs_find_qpus(start_node, partition, remote_wig)
                    
                    # Simple enumeration-based placement
                    enumeration_result = self.enumerate_find_qpus(partition)
                    
                    # Community-based placement
                    community_result, weighted_length_community = self.find_qpu_community(partition, remote_wig)
                    
                    # Store results
                    res[key].append(available_qpus)
                    res_community[key].append(community_result)
                    length[key] = weighted_length_bfs
                    length_community[key] = weighted_length_community
                    res_enumerate[key].append(enumeration_result)
                    
        return res, res_community, res_enumerate

    def bfs_find_qpus(self, start_node, partition, remote_wig):
        """
        Find a mapping from partitions to QPUs using breadth-first search.
        
        This method uses BFS to explore the QPU network and greedily assigns
        partitions to QPUs to minimize communication cost.
        
        Args:
            start_node: Starting QPU for BFS
            partition: Dictionary of partition assignments
            remote_wig: Remote weighted interaction graph
            
        Returns:
            Tuple of (mapping dictionary, weighted path length) or None if no mapping found
        """
        visited = set()
        count = Counter(partition)
        queue = [start_node]
        qpu_combination = {}
        
        # Sort partition requirements by size (largest first)
        distribution = list(count.values())
        distribution.sort(reverse=True)
        
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                qpu = self.network.nodes[node]['qpu']
                available_qubits = qpu.available_qubits
                
                # Try to assign partitions to this QPU
                for key, value in list(count.items()):  # Use list to avoid dictionary size change during iteration
                    if value <= available_qubits:
                        qpu_combination[key] = node
                        count.pop(key)
                        distribution.remove(value)
                        
                        # If all partitions are assigned, calculate the weighted length and return
                        if len(count.keys()) == 0:
                            original_sum, weighted_length = self.compute_weighted_length(
                                partition, remote_wig, qpu_combination)
                            return qpu_combination, weighted_length
                        break
                        
                # Explore neighbors
                for neighbor in self.network.neighbors(node):
                    if neighbor not in visited:
                        queue.append(neighbor)

        # No valid assignment found
        return None

    def enumerate_find_qpus(self, partition):
        """
        Find a mapping from partitions to QPUs by simple enumeration.
        
        This method simply enumerates all QPUs and assigns partitions to the
        QPUs with the most available qubits.
        
        Args:
            partition: Dictionary of partition assignments
            
        Returns:
            Mapping dictionary or None if no mapping found
        """
        # Get available qubits for each QPU
        qpus_available_qubits = {qpu.qpuid: qpu.available_qubits for qpu in self.qpus}
        count = Counter(partition)
        
        # Check if any QPU has enough qubits for the largest partition
        max_available_qubits = max(qpus_available_qubits.values())
        if max_available_qubits < max(count.values()):
            return None
            
        result = {}
        used_qpus = set()
        
        # Sort partitions by required qubits (largest first)
        for job_part, required_qubits in sorted(count.items(), key=lambda x: x[1], reverse=True):
            # Find QPUs with enough available qubits
            possible_qpus = [qpu for qpu in qpus_available_qubits.keys() 
                             if qpus_available_qubits[qpu] >= required_qubits]
            possible_qpus.sort(key=lambda x: qpus_available_qubits[x], reverse=True)
            
            # Try to find an unused QPU
            for qpu in possible_qpus:
                if qpu not in used_qpus:
                    result[job_part] = qpu
                    used_qpus.add(qpu)
                    break
                    
        return result

    def set_collaboration_data(self):
        """
        Set collaboration data for all QPUs in the cloud.
        
        This generates realistic physical parameters for the qubits in each QPU.
        """
        for i in range(len(self.qpus)):
            self.qpus[i].collaboration_data = self.generate_collboration_data(i)

    def generate_collboration_data(self, qpu_id):
        """
        Generate realistic physical parameters for qubits in a QPU.
        
        Creates a dataframe with parameters like T1, T2, frequencies, gate errors, etc.
        
        Args:
            qpu_id: ID of the QPU to generate data for
            
        Returns:
            Pandas DataFrame with qubit parameters
        """
        num_qubits = self.qpu_qubit_num
        
        # Define distributions for different parameters
        distributions = {
            "T1": {"mean": 100, "std": 20},  # Relaxation time (microseconds)
            "T2": {"mean": 150, "std": 30},  # Dephasing time (microseconds)
            "Frequency": {"mean": 5, "std": 0.1},  # Qubit frequency (GHz)
            "Anharmonicity": {"mean": -0.34, "std": 0.01},  # Anharmonicity (GHz)
            "Readout error": {"mean": 0.01, "std": 0.005},  # Readout error rate
            "CNOT error": {"mean": 0.005, "std": 0.001},  # CNOT gate error rate
            "Gate time": {"mean": 500, "std": 50},  # Gate time (nanoseconds)
        }
        
        # Generate data for each parameter
        data = {
            "Qubit": np.arange(num_qubits),
            "T1 (us)": np.random.normal(distributions["T1"]["mean"], distributions["T1"]["std"], num_qubits),
            "T2 (us)": np.random.normal(distributions["T2"]["mean"], distributions["T2"]["std"], num_qubits),
            "Frequency (GHz)": np.random.normal(distributions["Frequency"]["mean"], 
                                                distributions["Frequency"]["std"], num_qubits),
            "Anharmonicity (GHz)": np.random.normal(distributions["Anharmonicity"]["mean"],
                                                    distributions["Anharmonicity"]["std"], num_qubits),
            "Readout assignment error": np.random.normal(distributions["Readout error"]["mean"],
                                                         distributions["Readout error"]["std"], num_qubits),
        }
        
        # Generate two-qubit gate data
        cnot_errors = []
        gate_times = []
        for i in range(num_qubits):
            cnot_error = []
            gate_time = []
            for j in range(num_qubits):
                if i != j:  # No self-interaction
                    cnot_error.append(
                        f"{i}_{j}:{np.random.normal(distributions['CNOT error']['mean'], distributions['CNOT error']['std'])}")
                    gate_time.append(
                        f"{i}_{j}:{np.random.normal(distributions['Gate time']['mean'], distributions['Gate time']['std'])}")
            cnot_errors.append("; ".join(cnot_error))
            gate_times.append("; ".join(gate_time))

        data["CNOT error"] = cnot_errors
        data["Gate time (ns)"] = gate_times
        
        # Create a pandas DataFrame
        collaboration_data = pd.DataFrame(data)
        return collaboration_data


def create_random_topology(num_qpus, probability):
    return nx.erdos_renyi_graph(num_qpus, probability)

