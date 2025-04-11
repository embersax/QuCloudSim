# job scheduler define how to schedule job, should work with controoller
import random
import time
from job import job, job_generator
from cluster import qCloud, create_random_topology
from pytket import Circuit, OpType, qasm
import math
import networkx as nx
import pymetis
from des import Event, FinishedJob
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from pytket.utils import Graph
from flowScheduler import  flow_scheduler_1
from des import DES, generatingJob
from pytket.circuit.display import get_circuit_renderer

def generate_splits(total: int, parts: int) -> tuple:
    """
    Generate all possible ways to split a total number into a given number of parts.
    
    Args:
        total: Total number to split
        parts: Number of parts to split into
        
    Returns:
        Generator yielding tuples representing different splits
    """
    if parts == 1:
        yield (total,)
        return
        
    for i in range(total + 1):
        for split in generate_splits(total - i, parts - 1):
            yield (i,) + split


def max_scheduled_tasks_with_machines(jobs: list, machines: list) -> tuple:
    """
    Find maximum number of jobs that can be scheduled on given machines using dynamic programming.
    
    Args:
        jobs: List of job qubit requirements
        machines: List of machine qubit capacities
        
    Returns:
        tuple: (max_tasks, allocation_dict)
            - max_tasks: Maximum number of jobs that can be scheduled
            - allocation_dict: Dictionary mapping job indices to their allocations
    """
    n_jobs = len(jobs)
    n_machines = len(machines)
    
    # Initialize DP table and allocations tracking
    dp = np.zeros([n_jobs + 1] + [m + 1 for m in machines], dtype=int)
    shape = np.prod([m + 1 for m in machines])
    allocations = [[{} for _ in range(shape)] for _ in range(n_jobs + 1)]
    
    # Fill DP table
    for i in range(1, n_jobs + 1):
        job = jobs[i - 1]
        for res in np.ndindex(*[m + 1 for m in machines]):
            # Copy previous state
            dp[i][res] = dp[i - 1][res]
            idx = np.ravel_multi_index(res, [m + 1 for m in machines])
            allocations[i][idx] = allocations[i - 1][idx].copy()
            
            # Try different job splits across machines
            for split in generate_splits(job, n_machines):
                if all(res[j] >= split[j] for j in range(n_machines)):
                    new_res = tuple(res[j] - split[j] for j in range(n_machines))
                    new_idx = np.ravel_multi_index(new_res, [m + 1 for m in machines])
                    
                    if dp[i - 1][new_res] + 1 > dp[i][res]:
                        dp[i][res] = dp[i - 1][new_res] + 1
                        allocations[i][idx] = allocations[i - 1][new_idx].copy()
                        allocations[i][idx][i - 1] = split

    # Find maximum allocation
    max_tasks = 0
    final_res = tuple(machines)
    
    for res in np.ndindex(*[m + 1 for m in machines]):
        if dp[n_jobs][res] > max_tasks:
            max_tasks = dp[n_jobs][res]
            final_res = res

    # Create allocation dictionary
    final_idx = np.ravel_multi_index(final_res, [m + 1 for m in machines])
    final_alloc = allocations[n_jobs][final_idx]
    allocation_dict = {job_idx: alloc for job_idx, alloc in final_alloc.items()}

    return max_tasks, allocation_dict
  


class Placement:
    """
    Manages the placement and partitioning of quantum jobs across QPUs.
    
    This class handles the mapping of quantum circuits to physical QPUs,
    calculates communication costs, and manages circuit transformations
    for distributed execution.
    """
    
    def __init__(self, job, partition, qpu_mapping, wig):
        """
        Initialize a placement configuration.
        
        Args:
            job: Quantum job to be placed
            partition: Partition assignment (1 for single QPU, list for multiple)
            qpu_mapping: Mapping of partitions to QPUs
            wig: Weighted interaction graph of the circuit
        """
        self.job = job
        self.partition = partition if partition == 1 else partition[0]
        self.qpu_mapping = qpu_mapping
        self.wig = wig
        
        # Metrics and derived data
        self.communication_cost = None
        self.dag_longest_path_length = None
        self.score = None
        self.time = None
        self.modified_circuit = None
        self.remote_wig = None
        self.remote_dag = None
        self.improvement = False

    def get_time(self, partition):
        """
        Calculate execution time based on DAG longest path.
        
        Args:
            partition: Partition configuration to use for DAG creation
        """
        dag = self.get_remote_DAG(partition)
        longest_path = nx.dag_longest_path(dag)
        self.dag_longest_path_length = len(longest_path)
        self.time = len(longest_path)

    def get_remote_DAG(self, partition):
        """
        Create remote DAG representation of the circuit.
        
        Transforms the circuit into a DAG that represents remote operations
        between different partitions. Only includes gates that operate across
        partition boundaries.
        
        Args:
            partition: Partition configuration to use
            
        Returns:
            NetworkX DAG object representing remote operations
        """
        # Define single-qubit gates that don't need remote operations
        one_qubit_gates = [
            OpType.H, OpType.T, OpType.S, OpType.X, OpType.Y, OpType.Z,
            OpType.Rx, OpType.Ry, OpType.Rz, OpType.U1, OpType.U2, OpType.U3,
            OpType.Barrier
        ]
        
        # Initialize circuit and mappings
        circuit = self.job.circuit
        node_to_int = {node: i for i, node in enumerate(self.wig.nodes())}
        modified_circuit = Circuit()
        
        # Add qubits to modified circuit
        for qubit in circuit.qubits:
            modified_circuit.add_qubit(qubit)
            
        # Create partition mapping
        qubits = list(circuit.qubits)
        part_vert = partition[0]
        partitions = {i: [] for i in set(part_vert)}
        for i, part in enumerate(part_vert):
            partitions[part].append(qubits[i])
            
        # Create graph for remote operations
        graph = nx.Graph()
        
        # Process each gate in the circuit
        for command in circuit:
            op_type = command.op.type
            qubits = list(command.args)

            # Handle two-qubit gates (potential remote operations)
            if len(qubits) == 2 and op_type != OpType.Measure:
                node1, node2 = qubits
                part_1 = self.partition[node_to_int[node1]]
                part_2 = self.partition[node_to_int[node2]]
                
                # If qubits are in different partitions, add to remote circuit
                if part_1 != part_2:
                    modified_circuit.add_gate(op_type, qubits)
                    if not graph.has_edge(part_1, part_2):
                        graph.add_edge(part_1, part_2, weight=1)
                    else:
                        graph[part_1][part_2]['weight'] += 1
                        
        # Store results
        self.modified_circuit = modified_circuit
        self.remote_wig = graph

        # Create and return DAG
        g = Graph(modified_circuit)
        dag = g.as_nx()
        self.remote_dag = dag
        return dag

    def get_communication_cost(self, wig):
        """
        Calculate total communication cost based on edge weights between partitions.
        
        The communication cost is the sum of weights of edges that cross partition
        boundaries in the weighted interaction graph.
        
        Args:
            wig: Weighted interaction graph
        """
        total_cost = 0
        node_to_int = {node: i for i, node in enumerate(wig.nodes())}
        
        # Sum weights of edges between different partitions
        for node1, node2, data in wig.edges(data=True):
            if self.partition[node_to_int[node1]] != self.partition[node_to_int[node2]]:
                total_cost += data['weight']
                
        self.communication_cost = total_cost


class job_scheduler:
    """
    Job scheduler for quantum cloud computing.
    
    Responsible for scheduling quantum jobs on available QPUs in a quantum cloud.
    Different scheduling strategies can be employed based on scheduler_type.
    """
    
    def __init__(self, job_queue, des, qcloud, scheduler_type="default", flow_scheduler_type="default", logger=None):
        """
        Initialize the job scheduler.
        
        Args:
            job_queue: List of jobs to be scheduled
            des: Discrete Event Simulator instance
            qcloud: Quantum Cloud instance
            scheduler_type: Type of scheduling algorithm to use
            flow_scheduler_type: Type of flow scheduling algorithm to use
            logger: Logger for recording metrics
        """
        self.job_queue = job_queue
        self.qcloud = qcloud
        self.des = des
        self.scheduled_job = []
        # unscheduled_job stores the job that currently can't be processed
        self.unscheduled_job = []
        # registered_jobs stores the job that has been assigned placement and waiting for flow_scheduling
        self.registered_jobs = []
        self.schduler_type = scheduler_type
        self.flow_scheduler_type = flow_scheduler_type
        self.logger = logger
    
    def regiester_unscheduled_job(self):
        """
        Move all jobs from job_queue to unscheduled_job list.
        """
        while self.job_queue: 
            job = self.job_queue.pop(0)
            self.unscheduled_job.append(job)

    def schedule_choice(self):
        """
        Select and execute the appropriate scheduling strategy based on scheduler_type.
        """
        if self.schduler_type == "bfs":
            self.schedule_bfs()
        elif self.schduler_type == "greedy":
            self.schedule_greedy()
        elif self.schduler_type == "fifo":
            self.schedule_fifo()
        elif self.schduler_type == "annealing":
            self.schedule_annealing()
    

    def schedule_bfs(self):
        """
        BFS-based scheduling strategy.
        
        Uses breadth-first search for finding QPU placements.
        """
        # Sort jobs by a weighted combination of metrics
        self.job_queue.sort(
            key=lambda x: 0.3 * x.circuit.n_qubits + 
                          0.3 * (x.circuit.n_2qb_gates() / x.circuit.n_qubits) + 
                          0.4 * (x.circuit.depth()), 
            reverse=True
        )

        while len(self.job_queue) > 0:
            available_qubits = self.qcloud.get_available_qubits()
            
            # Check if all jobs need more qubits than available
            all_none = all([job.circuit.n_qubits > available_qubits for job in self.job_queue])
            if all_none:
                self.regiester_unscheduled_job()
                break
                
            job = self.job_queue[0]
            if job.circuit.n_qubits > available_qubits:
                self.unscheduled_job.append(self.job_queue.pop(0))
                print("no qpu available")
                continue
                
            possible_placements = self.find_simple_placement_bfs(job)
            if not possible_placements:
                self.regiester_unscheduled_job()
                break
                
            best_placement = self.score(possible_placements)
            print("find_placement")
            
            current_time = self.des.current_time
            best_placement.start_time = current_time
            self._schedule_new(best_placement, job, "BFS")
            
            self.scheduled_job.append(job)
            self.job_queue.pop(0)

        # Start flow scheduling
        flow_scheduler = flow_scheduler_1(self.des, self.qcloud, epr_p=0.3, name="BFS")
        flow_scheduler.run()
    
    def schedule_fifo(self):
        """
        First-In-First-Out scheduling strategy.
        
        Process jobs in order without reordering.
        """
        while len(self.job_queue) > 0:
            available_qubits = self.qcloud.get_available_qubits()
            
            # Check if all jobs need more qubits than available
            all_none = all([job.circuit.n_qubits > available_qubits for job in self.job_queue])
            if all_none:
                self.regiester_unscheduled_job()
                break
                
            job = self.job_queue[0]
            if job.circuit.n_qubits > available_qubits:
                self.unscheduled_job.append(self.job_queue.pop(0))
                print("no qpu available")
                continue
                
            possible_placements = self.find_simple_placement(job)
            if not possible_placements:
                self.unscheduled_job.append(self.job_queue.pop(0))
                continue

            best_placement = self.score(possible_placements)
            self._schedule_new(best_placement, job, "BFS")
            
            current_time = self.des.current_time
            best_placement.start_time = current_time
            self.scheduled_job.append(job)
            self.job_queue.pop(0)

            count = Counter(best_placement.partition)
            used_qpu_qubits = {value: count[key] for key, value in best_placement.qpu_mapping[0].items()}
            print("jobs are scheduled")
            
        # Start flow scheduling
        flow_scheduler = flow_scheduler_1(self.des, self.qcloud, epr_p=0.3, name="BFS")
        flow_scheduler.run()
        return

    def schedule_annealing(self):
        """
        Simulated Annealing-based scheduling strategy.
        
        Uses simulated annealing to find optimal placements.
        """
        print("annealing_start")
        while len(self.job_queue) > 0:
            available_qubits = self.qcloud.get_available_qubits()
            
            # Check if all jobs need more qubits than available
            all_none = all([job.circuit.n_qubits > available_qubits for job in self.job_queue])
            if all_none:
                self.regiester_unscheduled_job()
                break
                
            job = self.job_queue[0]
            if job.circuit.n_qubits > available_qubits:
                self.unscheduled_job.append(self.job_queue.pop(0))
                print("no qpu available")
                continue
                
            possible_placements = self.find_simple_placement_annealing(job)
            if not possible_placements:
                self.regiester_unscheduled_job()
                break

            best_placement = self.score(possible_placements)
            print(best_placement.partition)
            
            current_time = self.des.current_time
            best_placement.start_time = current_time
            self._schedule_new(best_placement, job, "BFS")
            
            self.scheduled_job.append(job)
            self.job_queue.pop(0)

        # Start flow scheduling
        flow_scheduler = flow_scheduler_1(self.des, self.qcloud, epr_p=0.3, name="BFS")
        flow_scheduler.run()
        return
    
    def schedule_greedy(self):
        """
        Greedy scheduling strategy.
        
        Attempts to maximize the number of jobs that can be scheduled.
        """
        qubits_requirement = [job.circuit.n_qubits for job in self.job_queue]
        qubits_qpu = [qpu.available_qubits for qpu in self.qcloud.qpus]
        selected_jobs = max_scheduled_tasks_with_machines(qubits_requirement, qubits_qpu)
        print(selected_jobs)
        # Note: This method appears to be incomplete in the original code
    
    def score(self, placement_list):
        """
        Score each placement and return the best one.
        
        Uses a weighted combination of execution time and communication cost.
        
        Args:
            placement_list: List of possible placements
            
        Returns:
            The placement with the highest score
        """
        if len(placement_list) == 1:
            return placement_list[0]
            
        time_list = [placement.time for placement in placement_list]
        communication_cost_list = [placement.communication_cost for placement in placement_list]
        
        min_time, max_time = min(time_list), max(time_list)
        min_communication_cost, max_communication_cost = min(communication_cost_list), max(communication_cost_list)
        
        # Calculate normalized inverse scores (lower values are better, so we invert)
        if min_time == max_time:
            inverse_normalized_time_list = [1 for _ in time_list]
        else:
            inverse_normalized_time_list = [1 - (time - min_time) / (max_time - min_time) for time in time_list]
            
        if min_communication_cost == max_communication_cost:
            inverse_normalized_communication_cost_list = [1 for _ in communication_cost_list]
        else:
            inverse_normalized_communication_cost_list = [
                        1 - (communication_cost - min_communication_cost) / (max_communication_cost - min_communication_cost) 
                        for communication_cost in communication_cost_list
                    ]
            
        # Compute the final score as weighted average
        for i, placement in enumerate(placement_list):
            placement.score = 0.5 * inverse_normalized_time_list[i] + \
                              0.5 * inverse_normalized_communication_cost_list[i]
        
        # Return the placement with highest score
        return max(placement_list, key=lambda x: x.score)
    
    def _schedule_new(self, placement, job, name=None):
        """
        Schedule a job with the given placement.
        
        Args:
            placement: Placement object with mapping information
            job: Job to schedule
            name: Name identifier for the scheduler
        """
        if placement.partition == 1:
            # Single QPU case
            selected_qpu = random.choice(placement.qpu_mapping)
            selected_qpu.allocate_job(job, job.circuit.n_qubits)
            used_qpu = {selected_qpu.qpuid: job.circuit.n_qubits}
            
            # Create finished event
            single_finished_event = FinishedJob(placement.time, job, used_qpu, str(job.id) + ' finished')
            self.des.schedule_event(single_finished_event)
            return
        
        # Multiple QPUs case
        counts = Counter(placement.partition)
        for par_idx, qpu_id in placement.qpu_mapping[0].items():
            # Allocate qubits on each QPU
            qpu = self.qcloud.network.nodes[qpu_id]['qpu']
            qpu.allocate_job(job, counts[par_idx])

        # Register job for flow scheduling
        self.registered_jobs.append((job, placement))
    
    def convert_to_weighted_graph(self, circuit):
        """
        Convert a quantum circuit to a weighted interaction graph.
        
        Nodes are qubits and edges represent interactions (gates) between qubits.
        Edge weights correspond to the number of interactions.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Weighted interaction graph
        """
        graph = nx.Graph()
        
        for command in circuit:
            op_type = command.op.type
            qubits = command.args
            
            # Add edge for two-qubit gates
            if op_type != OpType.Measure and len(qubits) == 2:
                q1, q2 = qubits
                if not graph.has_edge(q1, q2):
                    graph.add_edge(q1, q2, weight=1)
                else:
                    graph[q1][q2]['weight'] += 1
                    
        return graph
    
    def partition_circuit(self, n_parts, graph):
        """
        Partition the circuit graph into n_parts using PyMETIS.
        
        Args:
            n_parts: Number of partitions
            graph: Weighted interaction graph
            
        Returns:
            List of partitioning results
        """
        node_to_int = {node: i for i, node in enumerate(graph.nodes())}
        int_to_node = {i: node for node, i in node_to_int.items()}  # Reverse mapping

        xadj = [0]
        adjncy = []
        eweights = []

        # Prepare data for PyMETIS
        for node in graph.nodes():
            int_node = node_to_int[node]
            neighbors = [node_to_int[neighbor] for neighbor in graph.neighbors(node)]
            adjncy.extend(neighbors)
            xadj.append(xadj[-1] + len(neighbors))
            
            for int_neighbor in neighbors:
                orig_node = int_to_node[int_node]
                orig_neighbor = int_to_node[int_neighbor]
                if graph.has_edge(orig_node, orig_neighbor):
                    weight = graph[orig_node][orig_neighbor]['weight']
                    eweights.append(weight)
                else:
                    print(f"Edge not found: {orig_node} - {orig_neighbor}")
        
        ufactor_list = [20]  # Unbalance factor
        
        xadj = np.array(xadj)
        adjncy = np.array(adjncy)
        eweights = np.array(eweights)
        res = []
        
        for ufactor in ufactor_list:
            opt = pymetis.Options()
            opt.ufactor = ufactor
            cutcount, part_vert = pymetis.part_graph(n_parts, xadj=xadj, adjncy=adjncy, 
                                                     eweights=eweights, options=opt)
            res.append(part_vert)

        return res

    def find_simple_placement_bfs(self, job):
        """
        Find possible placements for a single job using BFS.
        
        Args:
            job: Job to find placements for
            
        Returns:
            List of possible placements
        """
        size = job.circuit.n_qubits
        circuit = job.circuit
        comb = {}
        wig = self.convert_to_weighted_graph(circuit)
        
        # Check if a single QPU is enough
        if size < max([qpu.available_qubits for qpu in self.qcloud.qpus]):
            single_qpu_list = [qpu for qpu in self.qcloud.qpus if qpu.available_qubits >= size]
            comb[1] = single_qpu_list
            
            res = Placement(job, 1, single_qpu_list, wig)
            res.communication_cost = 0
            
            single_graph = Graph(circuit)
            single_dag = single_graph.as_nx()
            res.time = len(nx.dag_longest_path(single_dag)) // 10
            res.modified_circuit = circuit

            return [res]
        else:
            # Try different numbers of partitions
            for i in range(math.ceil(size / self.qcloud.qpu_qubit_num),
                           min(len(self.qcloud.qpus), math.ceil(size / self.qcloud.qpu_qubit_num)) + 2):
                res = self.partition_circuit(i, wig)
                sublist_counts = [Counter(sublist) for sublist in res]
                comb[i] = res
                
            # Find placements for the partitions using BFS
            possible_placements = self.qcloud.find_placement_bfs(comb, wig)
            placement_list = []
            
            if not possible_placements:
                return None
                
            for key, value in possible_placements.items():
                if not value:
                    continue
                    
                all_none = all(x is None for x in value)
                if all_none:
                    continue
                    
                single_placement = Placement(job, comb[key], value, wig)
                single_placement.get_communication_cost(wig)
                single_placement.get_time(comb[key])
                placement_list.append(single_placement)
                
            return placement_list
    
    def find_simple_placement_annealing(self, job):
        """
        Find possible placements for a single job using simulated annealing.
        
        Args:
            job: Job to find placements for
            
        Returns:
            List of possible placements
        """
        size = job.circuit.n_qubits
        circuit = job.circuit
        comb = {}
        wig = self.convert_to_weighted_graph(circuit)
        
        # Check if a single QPU is enough
        if size < max([qpu.available_qubits for qpu in self.qcloud.qpus]):
            single_qpu_list = [qpu for qpu in self.qcloud.qpus if qpu.available_qubits >= size]
            comb[1] = single_qpu_list
            
            res = Placement(job, 1, single_qpu_list, wig)
            res.communication_cost = 0
            
            single_graph = Graph(circuit)
            single_dag = single_graph.as_nx()
            res.time = len(nx.dag_longest_path(single_dag)) // 10
            res.modified_circuit = circuit

            return [res]
        else:
            # Find placements using simulated annealing
            possible_placements, res = self.qcloud.sa_find_placement(job.circuit)
            placement_list = []
            
            if not possible_placements:
                return None
                
            for key, value in possible_placements.items():
                if not value:
                    continue
                    
                all_none = all(x is None for x in value)
                if all_none:
                    continue
                    
                single_placement = Placement(job, comb[key], value, wig)
                single_placement.get_communication_cost(wig)
                single_placement.get_time(comb[key])
                placement_list.append(single_placement)
                
            return placement_list

