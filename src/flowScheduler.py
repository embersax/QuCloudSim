# flow scheduler defines how to schedule quantum communication flows
import copy
import math
from itertools import combinations, permutations
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import random
from functools import lru_cache
import networkx as nx
from des import Event, FinishedJob
from utils import compute_probability, compute_probability_with_distance
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import concurrent.futures

# Set default EPR pair success probability
p = 0.4
sys.setrecursionlimit(1000000)


def _is_allocatable(to_go_pool, resources):
    """
    Check if any resources can still be allocated to any group in to_go_pool.
    
    Args:
        to_go_pool: Dictionary mapping QPU pairs to lists of nodes awaiting resources
        resources: Dictionary mapping QPU IDs to available communication qubits
        
    Returns:
        True if at least one QPU pair can still be allocated resources, False otherwise
    """
    for key, value in to_go_pool.items():
        qpu1, qpu2 = key
        if resources[qpu1] > 0 and resources[qpu2] > 0:
            return True

    return False


def allocate_resources_between_sets(tasks, machine_resources):
    """
    Allocate resources between different machine sets with equal priority.
    
    This function distributes available communication qubits among different QPU pairs,
    ensuring fair distribution when multiple tasks compete for the same resources.
    
    Args:
        tasks: Dictionary mapping machine sets (QPU pairs) to lists of tasks
        machine_resources: Dictionary mapping machines (QPUs) to available resources
        
    Returns:
        Dictionary mapping machine sets to their allocated resources
    """
    # Initialize the available resources for each machine set
    set_available_resources = {
        machine_set: min(machine_resources[machine] for machine in machine_set)
        for machine_set in tasks.keys() if tasks[machine_set]
    }

    # Initialize allocations
    allocations = {machine_set: 0 for machine_set in tasks.keys() if tasks[machine_set]}

    # Flag to indicate if we can still allocate resources
    can_allocate = True

    while can_allocate:
        can_allocate = False  # Assume no allocation is possible until proven otherwise

        for machine_set, jobs in tasks.items():
            if not jobs or set_available_resources[machine_set] <= 0:
                continue

            # Determine the allocation for this iteration
            job_count = len(jobs)
            allocation_per_job = min(1, set_available_resources[machine_set] / job_count)
            total_allocation = allocation_per_job * job_count

            if total_allocation > 0:
                can_allocate = True  # Allocation is possible, continue in the next iteration

                # Update allocations and available resources
                allocations[machine_set] += total_allocation
                for machine in machine_set:
                    machine_resources[machine] -= total_allocation

                # Update available resources for all sets sharing these machines
                for other_set in set_available_resources:
                    if any(machine in other_set for machine in machine_set):
                        set_available_resources[other_set] = min(machine_resources[m] for m in other_set)

    return allocations


def allocate_resources_random(to_go_pool, resources):
    """
    Randomly assign resources to nodes in the front layer when possible.
    
    This function allocates communication qubits to nodes based on random selection,
    without considering priorities or other factors.
    
    Args:
        to_go_pool: Dictionary mapping QPU pairs to lists of nodes awaiting resources
        resources: Dictionary mapping QPU IDs to available communication qubits
        
    Returns:
        Dictionary mapping nodes to their allocated resources
    """
    # Initialize result with zero allocations for all nodes
    allocation_result = {value: 0 for key, values in to_go_pool.items() for value in values}
    
    # Continue allocating while resources are available
    while _is_allocatable(to_go_pool, resources):
        # Randomly select a QPU pair that has nodes waiting
        key = random.choice(list(to_go_pool.keys()))
        # Randomly select a node from that pair
        node = random.choice(to_go_pool[key])
        # Allocate one resource to the node
        allocation_result[node] += 1
        # Reduce available resources
        resources[key[0]] -= 1
        resources[key[1]] -= 1
        
    return allocation_result


def allocate_resources_greedy(to_go_pool, resources, priority):
    """
    Allocate resources to nodes in the front layer based on node priorities.
    
    This function greedily allocates communication qubits to nodes with the highest
    priority first, maximizing resource utilization for important nodes.
    
    Args:
        to_go_pool: Dictionary mapping QPU pairs to lists of nodes awaiting resources
        resources: Dictionary mapping QPU IDs to available communication qubits
        priority: Dictionary mapping nodes to their priority values
        
    Returns:
        Dictionary mapping nodes to their allocated resources
    """
    # Initialize result with zero allocations for all nodes
    allocation_result = {value: 0 for key, values in to_go_pool.items() for value in values}
    allocated_node = set()  # Track nodes that have already been allocated resources
    
    # Continue allocating while resources are available
    while _is_allocatable(to_go_pool, resources):
        # Get all nodes that haven't been fully allocated yet
        candidates = [value for key, values in to_go_pool.items() for value in values if value not in allocated_node]
        
        if not candidates:
            break
            
        # Select the node with the highest priority
        most_important_node = max(candidates, key=lambda x: priority[x])
        
        # Find which QPU pair this node belongs to
        qpu1, qpu2 = next((qpu for qpu in to_go_pool.keys() if most_important_node in to_go_pool[qpu]), (None, None))
        
        if qpu1 is None or qpu2 is None:
            # This should not happen, but let's be safe
            allocated_node.add(most_important_node)
            continue
            
        # Calculate maximum available resources
        maximum_resources = min(resources[qpu1], resources[qpu2])
        
        # Allocate all available resources to this node
        allocation_result[most_important_node] += maximum_resources
        resources[qpu1] -= maximum_resources
        resources[qpu2] -= maximum_resources
        
        # Mark this node as allocated
        allocated_node.add(most_important_node)
        
    return allocation_result


def find_non_overlapping_indices(sets):
    """
    Find indices of sets that do not overlap with any other set.
    
    This function identifies quantum jobs that can be executed independently
    because they use different QPUs.
    
    Args:
        sets: List of sets where each set contains QPU IDs used by a job
        
    Returns:
        List of indices of non-overlapping sets
    """
    non_overlapping_indices = []

    for i, current_set in enumerate(sets):
        overlap = False
        for j, other_set in enumerate(sets):
            if i != j and not current_set.isdisjoint(other_set):
                overlap = True
                break
        if not overlap:
            non_overlapping_indices.append(i)

    return non_overlapping_indices


def group_tasks_by_overlap(tasks):
    """
    Group tasks based on resource overlap using a graph-based approach.
    
    This function creates groups of tasks that share resources, allowing for
    coordinated scheduling of tasks that compete for the same resources.
    
    Args:
        tasks: List of sets where each set contains resource IDs used by a task
        
    Returns:
        List of lists, where each inner list contains indices of tasks that overlap
    """
    # Create a graph where nodes are task indices
    G = nx.Graph()

    # Add nodes for each task
    for i in range(len(tasks)):
        G.add_node(i)

    # Add edges between nodes if their sets intersect
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            if tasks[i].intersection(tasks[j]):
                G.add_edge(i, j)

    # Find connected components (groups of overlapping tasks)
    groups = list(nx.connected_components(G))

    # Return the groups with indices
    result = []
    for group in groups:
        result.append(sorted(list(group)))

    return result


def find_two_farthest_parents(graph, target_node):
    """
    Find the two farthest ancestor nodes for a target node in a DAG.
    
    This function is used to identify the original qubits that interact through
    a two-qubit gate, by tracing back through the quantum circuit's DAG.
    
    Args:
        graph: NetworkX directed acyclic graph representing the quantum circuit
        target_node: Node for which to find the farthest parents
        
    Returns:
        List of farthest parent nodes
    """
    # Function to trace back the farthest parent for a specific port
    def trace_farthest_parent(node, port, visited):
        if node in visited:
            return node
        visited.add(node)

        for pred in graph.predecessors(node):
            edge_data_list = graph.get_edge_data(pred, node).values()
            for data in edge_data_list:
                if data['tgt_port'] == port:
                    return trace_farthest_parent(pred, data['src_port'], visited)
        return node

    # Find the two farthest ancestors based on the ports
    farthest_parents = set()
    for a, b, edge_data in graph.in_edges(target_node, data=True):
        farthest_parent = trace_farthest_parent(target_node, edge_data['tgt_port'], set())
        farthest_parents.add(farthest_parent)

    return list(farthest_parents)


class flow_scheduler_1:
    """
    Flow scheduler for distributed quantum computation.
    
    This class is responsible for scheduling quantum communication flows between
    QPUs when a quantum circuit is distributed across multiple QPUs.
    """
    
    def __init__(self, des, qcloud, epr_p, name):
        """
        Initialize the flow scheduler.
        
        Args:
            des: Discrete Event Simulator instance
            qcloud: Quantum Cloud instance
            epr_p: Success probability of EPR pair generation
            name: Identifier for this scheduler instance
        """
        self.des = des
        self.qcloud = qcloud
        self.EPR_p = epr_p
        self.name = name
    
    def run(self):
        """
        Execute the flow scheduling process for all registered jobs.
        
        This method identifies jobs that can be executed in parallel (using different QPUs)
        and those that need to be scheduled sequentially due to resource conflicts.
        """
        print("Start flow scheduling")
        
        # Get QPU mappings for all registered jobs
        qpu_mapping_list = [job[1].qpu_mapping[0] for job in self.des.scheduler.registered_jobs]
        qpu_set_list = [set(qpu_mapping.values()) for qpu_mapping in qpu_mapping_list]
        
        # Find the jobs that have no conflict with others
        non_conflict_job_indices = find_non_overlapping_indices(qpu_set_list)
        
        # Group tasks based on resource overlap
        grouped_tasks = group_tasks_by_overlap(qpu_set_list)
        print(grouped_tasks)
        
        # Sort groups by size (largest first for priority scheduling)
        grouped_tasks = sorted(grouped_tasks, key=lambda x: len(x), reverse=True)
        
        # Process each group
        for group in grouped_tasks:
            print(group)
            self.simulate_run_group(group)
            
        print("Finished flow scheduling")
        self.des.scheduler.registered_jobs.clear()

    def run_multi(self):
        """
        Execute the flow scheduling process using multi-processing.
        
        This method uses process pooling to execute multiple task groups concurrently.
        """
        print("Start flow scheduling with multi-processing")
        
        # Get QPU mappings for all registered jobs
        qpu_mapping_list = [job[1].qpu_mapping[0] for job in self.des.scheduler.registered_jobs]
        qpu_set_list = [set(qpu_mapping.values()) for qpu_mapping in qpu_mapping_list]

        # Find the jobs that have no conflict with others
        non_conflict_job_indices = find_non_overlapping_indices(qpu_set_list)

        # Group tasks based on resource overlap
        grouped_tasks = group_tasks_by_overlap(qpu_set_list)
        print(grouped_tasks)

        # Sort groups by size (smallest first for parallel execution)
        grouped_tasks = sorted(grouped_tasks, key=lambda x: len(x), reverse=False)

        # Use ProcessPoolExecutor for multi-processing
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit each group to be processed concurrently
            futures = [executor.submit(self.simulate_run_group, group) for group in grouped_tasks]

            # Wait for all futures to complete and get their results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        print("Finished flow scheduling with multi-processing")
        self.des.scheduler.registered_jobs.clear()
    
    def get_resources(self, togo_pool):
        """
        Get available communication resources for all QPUs in the pool.
        
        Args:
            togo_pool: Dictionary mapping job indices to QPU pairs with pending tasks
            
        Returns:
            Dictionary mapping QPU IDs to available communication qubits
        """
        # Collect all QPU IDs involved in any job
        single_key_list = []
        for key, value in togo_pool.items():
            for key1, value1 in value.items():
                single_key_list.append(key1)
                
        # Create a set of all unique QPU IDs
        result_set = set().union(*single_key_list)
        
        # Get available communication qubits for each QPU
        resources = {qpu_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits 
                     for qpu_idx in result_set}
        
        return resources
    
    def _is_allocatable(self, to_go_pool, resources):
        """
        Check if any resources can still be allocated to any group in to_go_pool.
        
        Args:
            to_go_pool: Dictionary mapping job indices to QPU pairs with pending tasks
            resources: Dictionary mapping QPU IDs to available communication qubits
            
        Returns:
            True if at least one QPU pair can still be allocated resources, False otherwise
        """
        for i, value in to_go_pool.items():
            for key in value.keys():
                qpu1, qpu2 = key
                if resources[qpu1] > 0 and resources[qpu2] > 0:
                    return True
        return False
    
    def check_competition_between_jobs(self, to_go_pool):
        """
        Identify competition for resources between different jobs.
        
        Args:
            to_go_pool: Dictionary mapping job indices to QPU pairs with pending tasks
            
        Returns:
            Dictionary mapping job index pairs to overlapping QPUs, or False if no competition
        """
        # Extract QPU sets for each job
        qpu_set_list = [list(job_dict.keys()) for job_dict in to_go_pool.values()]
        qpu_set_list = [set(qpu for qpu_pair in qpu_pairs for qpu in qpu_pair) 
                        for qpu_pairs in qpu_set_list]
        
        # Check for intersections between QPU sets
        competition_set = {}
        for (i, set1), (j, set2) in combinations(enumerate(qpu_set_list), 2):
            intersection = set1.intersection(set2)
            if intersection:
                if (i, j) not in competition_set:
                    competition_set[(i, j)] = intersection
        
        # Return False if no competition, otherwise return competition set
        if not competition_set:
            return False
        else:
            return competition_set
    
    def simulate_run_group(self, group, method=None):
        """
        Simulate the execution of a group of jobs that may have resource conflicts.
        
        This method handles the simulation of multiple jobs that share QPU resources,
        coordinating their execution and resource allocation.
        
        Args:
            group: List of indices of jobs to simulate
            method: Optional method specification to use for simulation
            
        Returns:
            None
        """
        # If there's only one job in the group, use the single job simulation
        if len(group) == 1:
            self._simulate_run_single(group)
            return
        
        # Initialize data structures for multiple job simulation
        graph_to_process = {i: copy.deepcopy(self.des.scheduler.registered_jobs[i][1].remote_dag) for i in group}
        priority = {i: self._compute_priority_1(graph_to_process[i]) for i in group}
        
        # Initialize tracking structures for each job
        input_layer = {}
        status_table = {}
        front_layer = {}
        output_layer = {}
        to_go_table = {}
        remote_combo = {}
        parents = {}
        parents_qpu = {}
        qpu_involved_list = {}
        
        # Process each job's circuit
        for i in group:
            placement = self.des.scheduler.registered_jobs[i][1]
            qpus_involved = set(placement.qpu_mapping[0].values())
            remote_dag = placement.remote_dag
            node_to_int = {str(node): i for i, node in enumerate(placement.wig.nodes())}
            
            # Identify input and output nodes
            input_layer[i] = [node for node in remote_dag.nodes() 
                          if remote_dag.nodes._nodes[node]['desc'] == 'Input']
            status_table[i] = {node: 0 for node in remote_dag.nodes()}
            
            # Remove input nodes from processing graph
            for node in input_layer[i]:
                graph_to_process[i].remove_node(node)

            # Identify and remove output nodes
            output_layer[i] = [node for node in remote_dag.nodes() 
                           if remote_dag.nodes._nodes[node]['desc'] == 'Output']
            for node in output_layer[i]:
                graph_to_process[i].remove_node(node)
            
            # Initialize tracking structures
            to_go_table[i] = {node: 'unready' for node in remote_dag.nodes()}
            qpu_list = sorted(list(qpus_involved))
            qpu_involved_list[i] = qpu_list
            remote_combo[i] = sorted(list(combinations(qpu_list, 2)))
            
            # Mark input nodes as finished
            for node in input_layer[i]:
                to_go_table[i][node] = 'finished'
            
            # Initialize parent tracking
            parents[i] = {node: [] for node in remote_dag.nodes()}
            parents_qpu[i] = {node: [] for node in remote_dag.nodes()}
            
            # Find parent nodes
            for node in graph_to_process[i].nodes():
                parents[i][node] = find_two_farthest_parents(remote_dag, node)
            
            # Map parents to QPUs
            data_type = placement.modified_circuit._dag_data[6]
            for node in parents[i].keys():
                qubit_index = [data_type[node] for node in parents[i][node]]
                if qubit_index:
                    parents[i][node] = [qubit_index]
                    parents_partition = [placement.partition[node_to_int[node]] for node in qubit_index]
                    parents_qpu[i][node] = sorted([placement.qpu_mapping[0][par] for par in parents_partition])

            # Remove nodes with both parents on same QPU
            nodes_to_delete = []
            for node in graph_to_process[i].nodes():
                if len(parents_qpu[i][node]) >= 2 and parents_qpu[i][node][0] == parents_qpu[i][node][1]:
                    nodes_to_delete.append(node)
                    to_go_table[i][node] = 'finished'
            for node in nodes_to_delete:
                graph_to_process[i].remove_node(node)

            # Initialize front layer
            front_layer[i] = [node for node in graph_to_process[i].nodes() 
                          if graph_to_process[i].in_degree(node) == 0 and 
                             graph_to_process[i].out_degree(node) != 0]
        
        print("Finished initialization")
        step = 0
        finished_group = set()
        
        # Print job names for logging
        job_names = [self.des.scheduler.registered_jobs[i][0].name for i in group]
        print(f"Starting flow scheduling for jobs: {job_names}")
        
        # Main simulation loop
        while any(len(graph_to_process[i].nodes()) != 0 for i in group):
            step += 1
            node_to_delete = {i: [] for i in group}
            node_to_add = {i: [] for i in group}
            to_go_pool = {i: {combo: [] for combo in remote_combo[i]} for i in group}
            
            # Group nodes by the QPU pairs they connect for each job
            for i in group:
                if i in finished_group:
                    continue
                    
                for node in front_layer[i]:
                    for combo in to_go_pool[i].keys():
                        if (parents_qpu[i][node] == list(combo) or 
                            parents_qpu[i][node] == list(combo)[::-1]):
                            to_go_pool[i][combo].append(node)

            # Remove empty entries
            to_go_pool = {i: {key: value for key, value in to_go_pool[i].items() if value} 
                          for i in group}
            
            # Initialize allocation tracking
            allocation_result = {i: {node: 0 for node in front_layer[i]} for i in group}
            
            # Check for competition between jobs
            competition_set = self.check_competition_between_jobs(to_go_pool)
            resources = self.get_resources(to_go_pool)

            # Resource allocation process
            if not competition_set:
                # No competition: allocate resources for each job independently
                for i in group:
                    if i in finished_group:
                        continue
                        
                    # Allocate resources between QPU pairs
                    inter_sets_allocations = allocate_resources_between_sets_priority(
                        to_go_pool[i], resources, priority[i])
                    
                    # Allocate resources within each QPU pair
                    for comb, nodes in to_go_pool[i].items():
                        if nodes:
                            total_resources = inter_sets_allocations[comb]
                            total_priority = sum([priority[i][node] for node in nodes])
                            
                            # First allocation round based on proportional priorities
                            for node in nodes:
                                if total_priority > 0:  # Avoid division by zero
                                    allocation_result[i][node] = round(total_resources * priority[i][node] / total_priority)
                                total_resources -= allocation_result[i][node]
                            
                            # Distribute remaining resources
                            if total_resources > 0:
                                sorted_nodes = sorted(nodes, key=lambda n: priority[i][n], reverse=True)
                                for node in sorted_nodes:
                                    if total_resources > 0:
                                        allocation_result[i][node] += 1
                                        total_resources -= 1
            else:
                # Competition exists: allocate resources across jobs by priority
                while self._is_allocatable(to_go_pool, resources):
                    print("Allocating resources with competition")
                    
                    for i in group:
                        if i in finished_group:
                            continue
                            
                        # Find candidates for allocation
                        candidates = []
                        allocation_level = 0
                                          
                        # Prioritize nodes with lower current allocation
                        max_allocation = max(allocation_result[i].values()) if allocation_result[i] else 0
                        while not candidates and allocation_level <= max_allocation:
                            candidates = [
                                node for node in front_layer[i]
                              if allocation_result[i][node] == allocation_level
                                and any(resources[q1] > 0 and resources[q2] > 0 
                                       for q1, q2 in to_go_pool[i].keys() 
                                       if node in to_go_pool[i][(q1, q2)])
                            ]
                            allocation_level += 1
                        
                        # Allocate resources to the highest priority candidate
                        allocated = False
                        while candidates and not allocated:
                            chosen_node = max(candidates, key=lambda n: priority[i][n])
                            qpus = next((key for key, value in to_go_pool[i].items() 
                                       if chosen_node in value), None)
                            
                            if qpus and resources[qpus[0]] > 0 and resources[qpus[1]] > 0:
                                allocation_result[i][chosen_node] += 1
                                resources[qpus[0]] -= 1
                                resources[qpus[1]] -= 1
                                allocated = True
                            else:
                                candidates.remove(chosen_node)

            # Process the simulation step for each job
            for i in group:
                if i in finished_group or len(graph_to_process[i].nodes()) == 0:
                    continue
                    
                # Process each node in the front layer
                for node in front_layer[i]:
                    if allocation_result[i][node] > 0:
                        # Calculate success probability based on resources and distance
                        distance = nx.shortest_path_length(
                            self.qcloud.network,
                            parents_qpu[i][node][0],
                            parents_qpu[i][node][1]
                        )
                        p_topo = compute_probability_with_distance(
                            allocation_result[i][node], 
                            self.EPR_p, 
                            distance
                        )
                        
                        # Check if the gate succeeds
                        seed = random.uniform(0, 1)
                        if seed >= 1 - p_topo:
                            status_table[i][node] = step
                            node_to_delete[i].append(node)
                            to_go_table[i][node] = 'finished'
                            
                            # Check if any successors are now ready
                            for succ in graph_to_process[i].succ[node]:
                                if all([to_go_table[i][pred] == 'finished' 
                                       for pred in graph_to_process[i].pred[succ]]):
                                    to_go_table[i][succ] = 'ready'
                                    if node not in output_layer[i]:
                                        node_to_add[i].append(succ)
                
                # Update the graph and front layer
                for node in node_to_delete[i]:
                    front_layer[i].remove(node)
                    graph_to_process[i].remove_node(node)
                for node in node_to_add[i]:
                    front_layer[i].append(node)

                # Check if job is completed
                if len(graph_to_process[i].nodes()) == 0:
                    current_time = self.des.current_time + step
                    count = Counter(self.des.scheduler.registered_jobs[i][1].partition)
                    qpu_mapping = self.des.scheduler.registered_jobs[i][1].qpu_mapping[0]
                    used_qpu = {value: count[key] for key, value in qpu_mapping.items()}
                    
                    # Create finished job event
                    finished_event = FinishedJob(
                        current_time, 
                        self.des.scheduler.registered_jobs[i][0], 
                        used_qpu,
                        f"{self.des.scheduler.registered_jobs[i][0].id} finished+withallocation{self.name}",
                        self.des.scheduler.registered_jobs[i][1]
                    )
                    self.des.schedule_event(finished_event)
                    finished_group.add(i)
                    print(f'Finished job {i}')

        print("All jobs in group completed")
        return 
        


    def _simulate_run_single_average(self, group):
        """
        Simulate running a single job using average resource allocation strategy.
        
        This method implements a resource allocation strategy where resources are
        distributed evenly among all competing nodes in the front layer.
        
        Args:
            group: List containing a single job index to simulate
            
        Returns:
            int: Number of steps taken to complete the simulation
        """
        current_step = 0
        placement = self.des.scheduler.registered_jobs[group[0]][1]
        remote_dag = placement.remote_dag
        node_to_int = {str(node): i for i, node in enumerate(placement.wig.nodes())}
        
        # Get the set of QPUs involved in this job
        qpus_involved = set(placement.qpu_mapping[0].values())
        used_qpus = sorted(list(qpus_involved))
        remote_combo = list(combinations(used_qpus, 2))
        
        # Initialize graph for processing
        graph_to_process = copy.deepcopy(remote_dag)
        
        # Remove input and output nodes
        input_layer = [node for node in remote_dag.nodes() 
                    if remote_dag.nodes._nodes[node]['desc'] == 'Input']
        output_layer = [node for node in remote_dag.nodes() 
                    if remote_dag.nodes._nodes[node]['desc'] == 'Output']
        
        for node in input_layer + output_layer:
            if node in graph_to_process.nodes():
                graph_to_process.remove_node(node)
        
        # Initialize tracking structures
        status_table = {node: 0 for node in remote_dag.nodes()}
        to_go_table = {node: 'unready' for node in remote_dag.nodes()}
        for node in input_layer:
            to_go_table[node] = 'finished'
        
        # Find parent nodes and their QPUs
        parents = {node: [] for node in graph_to_process.nodes()}
        parents_qpu = {node: [] for node in graph_to_process.nodes()}
        
        for node in graph_to_process.nodes():
            parents[node] = find_two_farthest_parents(remote_dag, node)
        
        data_type = placement.modified_circuit._dag_data[6]
        for node in parents.keys():
            qubit_index = [data_type[node] for node in parents[node]]
            parents[node] = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_partition = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_qpu[node] = sorted([placement.qpu_mapping[0][par] for par in parents_partition])
        
        # Remove nodes with parents on same QPU
        nodes_to_delete = []
        for node in graph_to_process.nodes():
            if parents_qpu[node][0] == parents_qpu[node][1]:
                nodes_to_delete.append(node)
                to_go_table[node] = 'finished'
        for node in nodes_to_delete:
            graph_to_process.remove_node(node)
        
        # Initialize front layer
        front_layer = [node for node in graph_to_process.nodes() 
                    if graph_to_process.in_degree(node) == 0 and 
                        graph_to_process.out_degree(node) != 0]
        
        print(f"Starting average allocation simulation for job: {self.des.scheduler.registered_jobs[group[0]][0].name}")
        
        # Main simulation loop
        while len(graph_to_process.nodes()) != 0:
            current_step += 1
            node_to_delete = []
            node_to_add = []
            
            # Group nodes by QPU pairs
            to_go_pool = {combo: [] for combo in remote_combo}
            for node in front_layer:
                for combo in to_go_pool.keys():
                    if parents_qpu[node] == list(combo) or parents_qpu[node] == list(combo)[::-1]:
                        to_go_pool[combo].append(node)
            
            # Remove empty entries
            to_go_pool = {key: value for key, value in to_go_pool.items() if value}
            
            # Calculate available resources per QPU
            resources = {qpu_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits 
                        for par_idx, qpu_idx in placement.qpu_mapping[0].items()}
            
            # Distribute resources evenly among nodes
            allocation_result = {node: 0 for node in front_layer}
            for combo, nodes in to_go_pool.items():
                if nodes:
                    total_resources = min(resources[combo[0]], resources[combo[1]])
                    resources_per_node = total_resources // len(nodes)
                    remaining_resources = total_resources % len(nodes)
                    
                    # Allocate base resources
                    for node in nodes:
                        allocation_result[node] = resources_per_node
                    
                    # Distribute remaining resources
                    for i in range(remaining_resources):
                        allocation_result[nodes[i]] += 1
            
            # Process nodes with allocated resources
            for node in front_layer:
                if allocation_result[node] > 0:
                    # Calculate success probability
                    distance = nx.shortest_path_length(
                        self.qcloud.network,
                        parents_qpu[node][0],
                        parents_qpu[node][1]
                    )
                    p_topo = compute_probability_with_distance(
                        allocation_result[node],
                        self.EPR_p,
                        distance
                    )
                    
                    # Check if operation succeeds
                    if random.uniform(0, 1) >= 1 - p_topo:
                        status_table[node] = current_step
                        node_to_delete.append(node)
                        to_go_table[node] = 'finished'
                        
                        # Check successors
                        for succ in graph_to_process.succ[node]:
                            if all([to_go_table[pred] == 'finished' 
                                for pred in graph_to_process.pred[succ]]):
                                to_go_table[succ] = 'ready'
                                if node not in output_layer:
                                    node_to_add.append(succ)
            
            # Update graph and front layer
            for node in node_to_delete:
                front_layer.remove(node)
                graph_to_process.remove_node(node)
            for node in node_to_add:
                front_layer.append(node)
        
        # Create completion event
        current_time = self.des.current_time + current_step
        count = Counter(placement.partition)
        used_qpu = {value: count[key] for key, value in placement.qpu_mapping[0].items()}
        
        finished_event = FinishedJob(
            current_time,
            self.des.scheduler.registered_jobs[group[0]][0],
            used_qpu,
            f"{self.des.scheduler.registered_jobs[group[0]][0].id} finished+withallocation{self.name}",
            self.des.scheduler.registered_jobs[group[0]][1]
        )
        self.des.schedule_event(finished_event)
        
        return current_step

    def _simulate_run_single_random(self, group):
        """
        Simulate running a single job using random resource allocation strategy.
        
        This method implements a resource allocation strategy where resources are
        distributed randomly among competing nodes in the front layer.
        
        Args:
            group: List containing a single job index to simulate
            
        Returns:
            int: Number of steps taken to complete the simulation
        """
        current_step = 0
        placement = self.des.scheduler.registered_jobs[group[0]][1]
        remote_dag = placement.remote_dag
        node_to_int = {str(node): i for i, node in enumerate(placement.wig.nodes())}
        
        # Setup is similar to average method
        qpus_involved = set(placement.qpu_mapping[0].values())
        used_qpus = sorted(list(qpus_involved))
        remote_combo = list(combinations(used_qpus, 2))
        
        graph_to_process = copy.deepcopy(remote_dag)
        
        # Remove input and output nodes
        input_layer = [node for node in remote_dag.nodes() 
                    if remote_dag.nodes._nodes[node]['desc'] == 'Input']
        output_layer = [node for node in remote_dag.nodes() 
                    if remote_dag.nodes._nodes[node]['desc'] == 'Output']
        
        for node in input_layer + output_layer:
            if node in graph_to_process.nodes():
                graph_to_process.remove_node(node)
        
        # Initialize tracking structures
        status_table = {node: 0 for node in remote_dag.nodes()}
        to_go_table = {node: 'unready' for node in remote_dag.nodes()}
        for node in input_layer:
            to_go_table[node] = 'finished'
        
        # Process parent nodes
        parents = {node: [] for node in graph_to_process.nodes()}
        parents_qpu = {node: [] for node in graph_to_process.nodes()}
        
        for node in graph_to_process.nodes():
            parents[node] = find_two_farthest_parents(remote_dag, node)
        
        data_type = placement.modified_circuit._dag_data[6]
        for node in parents.keys():
            qubit_index = [data_type[node] for node in parents[node]]
            parents[node] = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_partition = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_qpu[node] = sorted([placement.qpu_mapping[0][par] for par in parents_partition])
        
        # Remove nodes with parents on same QPU
        nodes_to_delete = []
        for node in graph_to_process.nodes():
            if parents_qpu[node][0] == parents_qpu[node][1]:
                nodes_to_delete.append(node)
                to_go_table[node] = 'finished'
        for node in nodes_to_delete:
            graph_to_process.remove_node(node)
        
        # Initialize front layer
        front_layer = [node for node in graph_to_process.nodes() 
                    if graph_to_process.in_degree(node) == 0 and 
                        graph_to_process.out_degree(node) != 0]
        
        print(f"Starting random allocation simulation for job: {self.des.scheduler.registered_jobs[group[0]][0].name}")
        
        # Main simulation loop
        while len(graph_to_process.nodes()) != 0:
            current_step += 1
            node_to_delete = []
            node_to_add = []
            
            # Group nodes by QPU pairs
            to_go_pool = {combo: [] for combo in remote_combo}
            for node in front_layer:
                for combo in to_go_pool.keys():
                    if parents_qpu[node] == list(combo) or parents_qpu[node] == list(combo)[::-1]:
                        to_go_pool[combo].append(node)
            
            # Remove empty entries
            to_go_pool = {key: value for key, value in to_go_pool.items() if value}
            
            # Calculate available resources
            resources = {qpu_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits 
                        for par_idx, qpu_idx in placement.qpu_mapping[0].items()}
            
            # Random resource allocation
            allocation_result = {node: 0 for node in front_layer}
            for combo, nodes in to_go_pool.items():
                if nodes:
                    total_resources = min(resources[combo[0]], resources[combo[1]])
                    while total_resources > 0:
                        # Randomly select a node to receive a resource
                        node = random.choice(nodes)
                        allocation_result[node] += 1
                        total_resources -= 1
            
            # Process nodes with allocated resources
            for node in front_layer:
                if allocation_result[node] > 0:
                    distance = nx.shortest_path_length(
                        self.qcloud.network,
                        parents_qpu[node][0],
                        parents_qpu[node][1]
                    )
                    p_topo = compute_probability_with_distance(
                        allocation_result[node],
                        self.EPR_p,
                        distance
                    )
                    
                    if random.uniform(0, 1) >= 1 - p_topo:
                        status_table[node] = current_step
                        node_to_delete.append(node)
                        to_go_table[node] = 'finished'
                        
                        for succ in graph_to_process.succ[node]:
                            if all([to_go_table[pred] == 'finished' 
                                for pred in graph_to_process.pred[succ]]):
                                to_go_table[succ] = 'ready'
                                if node not in output_layer:
                                    node_to_add.append(succ)
            
            # Update graph and front layer
            for node in node_to_delete:
                front_layer.remove(node)
                graph_to_process.remove_node(node)
            for node in node_to_add:
                front_layer.append(node)
        
        # Create completion event
        current_time = self.des.current_time + current_step
        count = Counter(placement.partition)
        used_qpu = {value: count[key] for key, value in placement.qpu_mapping[0].items()}
        
        finished_event = FinishedJob(
            current_time,
            self.des.scheduler.registered_jobs[group[0]][0],
            used_qpu,
            f"{self.des.scheduler.registered_jobs[group[0]][0].id} finished+withallocation{self.name}",
            self.des.scheduler.registered_jobs[group[0]][1]
        )
        self.des.schedule_event(finished_event)
        
        return current_step

    def _simulate_run_single_greedy(self, group):
        """
        Simulate running a single job using greedy resource allocation strategy.
        
        This method implements a greedy resource allocation strategy where resources
        are allocated to nodes with the highest priority until exhausted.
        
        Args:
            group: List containing a single job index to simulate
            
        Returns:
            int: Number of steps taken to complete the simulation
        """
        current_step = 0
        placement = self.des.scheduler.registered_jobs[group[0]][1]
        remote_dag = placement.remote_dag
        node_to_int = {str(node): i for i, node in enumerate(placement.wig.nodes())}
        
        # Setup is similar to other methods
        qpus_involved = set(placement.qpu_mapping[0].values())
        used_qpus = sorted(list(qpus_involved))
        remote_combo = list(combinations(used_qpus, 2))
        
        graph_to_process = copy.deepcopy(remote_dag)
        
        # Calculate node priorities
        priority = self._compute_priority_1(graph_to_process)
        
        # Remove input and output nodes
        input_layer = [node for node in remote_dag.nodes() 
                    if remote_dag.nodes._nodes[node]['desc'] == 'Input']
        output_layer = [node for node in remote_dag.nodes() 
                    if remote_dag.nodes._nodes[node]['desc'] == 'Output']
        
        for node in input_layer + output_layer:
            if node in graph_to_process.nodes():
                graph_to_process.remove_node(node)
        
        # Initialize tracking structures
        status_table = {node: 0 for node in remote_dag.nodes()}
        to_go_table = {node: 'unready' for node in remote_dag.nodes()}
        for node in input_layer:
            to_go_table[node] = 'finished'
        
        # Process parent nodes
        parents = {node: [] for node in graph_to_process.nodes()}
        parents_qpu = {node: [] for node in graph_to_process.nodes()}
        
        for node in graph_to_process.nodes():
            parents[node] = find_two_farthest_parents(remote_dag, node)
        
        data_type = placement.modified_circuit._dag_data[6]
        for node in parents.keys():
            qubit_index = [data_type[node] for node in parents[node]]
            parents[node] = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_partition = [placement.partition[node_to_int[node]] for node in qubit_index]
            parents_qpu[node] = sorted([placement.qpu_mapping[0][par] for par in parents_partition])
        
        # Remove nodes with parents on same QPU
        nodes_to_delete = []
        for node in graph_to_process.nodes():
            if parents_qpu[node][0] == parents_qpu[node][1]:
                nodes_to_delete.append(node)
                to_go_table[node] = 'finished'
        for node in nodes_to_delete:
            graph_to_process.remove_node(node)
        
        # Initialize front layer
        front_layer = [node for node in graph_to_process.nodes() 
                    if graph_to_process.in_degree(node) == 0 and 
                        graph_to_process.out_degree(node) != 0]
        
        print(f"Starting greedy allocation simulation for job: {self.des.scheduler.registered_jobs[group[0]][0].name}")
        
        # Main simulation loop
        while len(graph_to_process.nodes()) != 0:
            current_step += 1
            node_to_delete = []
            node_to_add = []
            
            # Group nodes by QPU pairs
            to_go_pool = {combo: [] for combo in remote_combo}
            for node in front_layer:
                for combo in to_go_pool.keys():
                    if parents_qpu[node] == list(combo) or parents_qpu[node] == list(combo)[::-1]:
                        to_go_pool[combo].append(node)
            
            # Remove empty entries
            to_go_pool = {key: value for key, value in to_go_pool.items() if value}
            
            # Calculate available resources
            resources = {qpu_idx: self.qcloud.network.nodes[qpu_idx]['qpu'].ncm_qubits 
                        for par_idx, qpu_idx in placement.qpu_mapping[0].items()}
            
            # Greedy resource allocation
            allocation_result = {node: 0 for node in front_layer}
            for combo, nodes in to_go_pool.items():
                if nodes:
                    total_resources = min(resources[combo[0]], resources[combo[1]])
                    # Sort nodes by priority
                    sorted_nodes = sorted(nodes, key=lambda n: priority[n], reverse=True)
                    
                    # Allocate all resources to highest priority nodes first
                    for node in sorted_nodes:
                        if total_resources > 0:
                            allocation_result[node] = total_resources
                            total_resources = 0
            
            # Process nodes with allocated resources
            for node in front_layer:
                if allocation_result[node] > 0:
                    distance = nx.shortest_path_length(
                        self.qcloud.network,
                        parents_qpu[node][0],
                        parents_qpu[node][1]
                    )
                    p_topo = compute_probability_with_distance(
                        allocation_result[node],
                        self.EPR_p,
                        distance
                    )
                    
                    if random.uniform(0, 1) >= 1 - p_topo:
                        status_table[node] = current_step
                        node_to_delete.append(node)
                        to_go_table[node] = 'finished'
                        
                        for succ in graph_to_process.succ[node]:
                            if all([to_go_table[pred] == 'finished' 
                                for pred in graph_to_process.pred[succ]]):
                                to_go_table[succ] = 'ready'
                                if node not in output_layer:
                                    node_to_add.append(succ)
            
            # Update graph and front layer
            for node in node_to_delete:
                front_layer.remove(node)
                graph_to_process.remove_node(node)
            for node in node_to_add:
                front_layer.append(node)
        
        # Create completion event
        current_time = self.des.current_time + current_step
        count = Counter(placement.partition)
        used_qpu = {value: count[key] for key, value in placement.qpu_mapping[0].items()}
        
        finished_event = FinishedJob(
            current_time,
            self.des.scheduler.registered_jobs[group[0]][0],
            used_qpu,
            f"{self.des.scheduler.registered_jobs[group[0]][0].id} finished+withallocation{self.name}",
            self.des.scheduler.registered_jobs[group[0]][1]
        )
        self.des.schedule_event(finished_event)
        
        return current_step