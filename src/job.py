"""
This module defines the job submission system for the quantum cloud simulator.
It includes classes for generating quantum jobs with different circuit types and
managing job queues.
"""
import os
import random
import numpy as np
from pytket import Circuit, OpType, qasm
import multiprocessing

# Global job ID counter to assign unique IDs to jobs
global current_job_id
current_job_id = 0


class job_generator:
    """
    Responsible for generating various quantum jobs with different circuit types.
    Can create jobs from existing QASM files or generate them with specific properties.
    """
    def __init__(self):
        """
        Initialize the job generator with paths to different circuit directories.
        Maps circuit categories to their file paths for easy access.
        """
        # 可以通过环境变量或配置文件来设置基础路径
        self.base_path = os.path.join(os.path.dirname(__file__), "..", "circuits")
        
        self.circuit_paths = {
            'small': self.get_qasm_files(os.path.join(self.base_path, "small")),
            'medium': self.get_qasm_files(os.path.join(self.base_path, "medium")),
            'large': self.get_qasm_files(os.path.join(self.base_path, "large")),
            'fixed': self.get_qasm_files(os.path.join(self.base_path, "test_whole")),
            'all_pool': self.get_qasm_files(os.path.join(self.base_path, "pool_2")),
            'qft': self.get_qasm_files(os.path.join(self.base_path, "pool_qft")),
            'qugan': self.get_qasm_files(os.path.join(self.base_path, "pool_qugan")),
            'arith': self.get_qasm_files(os.path.join(self.base_path, "pool_arith"))
        }

    def generate_circuit_arith_pool(self):
        """
        Generate a circuit from the arithmetic circuit pool.
        Returns the circuit name, circuit object, shot count, and arrival time.
        """
        picked_circuit = self.select_random_circuit('arith')
        name = picked_circuit.split('/')[-1]
        print(name)
        try:
            circuit = qasm.circuit_from_qasm(picked_circuit)
        except:
            print(f"Error loading circuit: {picked_circuit}")
        shots = random.uniform(50, 1000)
        # Time set to 0 for now, could represent arrival time
        time = 0
        return name, circuit, shots, time
    
    def generate_circuit_qft_pool(self):
        """
        Generate a circuit from the QFT circuit pool.
        Returns the circuit name, circuit object, shot count, and arrival time.
        """
        picked_circuit = self.select_random_circuit('qft')
        name = picked_circuit.split('/')[-1]
        print(name)
        try:
            circuit = qasm.circuit_from_qasm(picked_circuit)
        except:
            print(f"Error loading circuit: {picked_circuit}")
        shots = random.uniform(50, 1000)
        time = 0
        return name, circuit, shots, time
    
    def generate_circuit_qugan_pool(self):
        """
        Generate a circuit from the QuGAN circuit pool.
        Returns the circuit name, circuit object, shot count, and arrival time.
        """
        picked_circuit = self.select_random_circuit('qugan')
        name = picked_circuit.split('/')[-1]
        print(name)
        try:
            circuit = qasm.circuit_from_qasm(picked_circuit)
        except:
            print(f"Error loading circuit: {picked_circuit}")
        shots = random.uniform(50, 1000)
        time = 0
        return name, circuit, shots, time

    def generate_circuit_from_dirctory(self, directory):
        """
        Generate jobs from all QASM files in a specified directory.
        Returns a list of job objects.
        """
        qasm_files = self.get_qasm_files(directory)
        job_list = []
        for file in qasm_files:
            circuit = qasm.circuit_from_qasm(file)
            shots = random.uniform(50, 1000)
            time = 0  # Set to 0 for deterministic behavior
            name = file.split('/')[-1]
            job_list.append(job(name, circuit, shots, time))
        return job_list
         
    def get_qasm_files(self, directory):
        """
        Recursively find all QASM files in a directory.
        Returns a list of full file paths.
        """
        qasm_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.qasm'):
                    qasm_files.append(os.path.join(root, file))
        return qasm_files

    def select_random_circuit(self, directory):
        """
        Select a random circuit from the specified directory category.
        """
        return random.choice(self.circuit_paths[directory]) if self.circuit_paths[directory] else None

    def generate_fixed_job(self, path):
        """
        Generate a job from a specific QASM file path.
        """
        name = path.split('/')[-1]
        circuit = qasm.circuit_from_qasm(path)
        shots = 1000
        time = 0
        return job(name, circuit, shots, time)

    def generate_circuit_fixed_pool(self):
        """
        Generate a circuit from the fixed circuit pool.
        Returns the circuit name, circuit object, shot count, and arrival time.
        """
        picked_circuit = self.select_random_circuit('fixed')
        name = picked_circuit.split('/')[-1]
        try:
            circuit = qasm.circuit_from_qasm(picked_circuit)
        except:
            print(f"Error loading circuit: {picked_circuit}")
        shots = random.uniform(50, 1000)
        time = 0
        return name, circuit, shots, time

    def generate_circuit(self, probability):
        """
        Generate a circuit based on probability distribution for small/medium/large circuits.
        
        Args:
            probability: List of probabilities for [small, medium, large] circuit selection
            
        Returns:
            circuit: The generated quantum circuit
            shots: Random number of shots for execution
            time: Random arrival time (currently set to distribute in 0-10 range)
        """
        # Normalize probabilities if they don't sum to 1
        total_prob = sum(probability)
        if total_prob != 1:
            probability = [p / total_prob for p in probability]

        # Generate a random number using a uniform distribution
        distribution = np.random.uniform(0, 1)

        # Decide which circuit to use based on the probabilities
        if distribution < probability[0]:  # Probability for small circuit
            circuit = qasm.circuit_from_qasm(self.select_random_circuit('small'))
        elif distribution < probability[0] + probability[1]:  # Probability for medium circuit
            circuit = qasm.circuit_from_qasm(self.select_random_circuit('medium'))
        else:  # Remaining probability for large circuit
            circuit = qasm.circuit_from_qasm(self.select_random_circuit('large'))

        # Generate random number of shots and arrival time
        shots = random.uniform(50, 1000)
        time = random.uniform(0, 10)
        return circuit, shots, time

    def generate_large_circuit_job(self):
        """
        Generate a job with a large circuit for testing purposes.
        """
        circuit = qasm.circuit_from_qasm(self.select_random_circuit('large'))
        shots = random.uniform(50, 1000)
        time = random.uniform(0, 10)
        return job(circuit, shots, time)

    def get_file_number(self, directory):
        """
        Get the number of QASM files in a directory.
        """
        return len(self.get_qasm_files(directory))

    def generate_job(self, n, time_frame=0, step=0, probability=[0.33, 0.33, 0.34]):
        """
        Generate n jobs with given parameters.
        
        Args:
            n: Number of jobs to generate
            time_frame: Time frame for each step
            step: Current step number
            probability: Probability distribution for circuit sizes [small, medium, large]
            
        Returns:
            A list of job objects
        """
        jobs = []
        for i in range(n):
            circuit, shots, time = self.generate_circuit(probability)
            jobs.append(job(f"job_{i}", circuit, shots, time + time_frame * step))
        return jobs

    def generate_job_fixed_pool(self, n, time_frame, step):
        """
        Generate n jobs from the fixed circuit pool.
        """
        jobs = []
        for i in range(n):
            name, circuit, shots, time = self.generate_circuit_fixed_pool()
            jobs.append(job(name, circuit, shots, time + time_frame * step))
        return jobs
        
    def generate_job_qft_pool(self, n, time_frame, step):
        """
        Generate n jobs from the QFT circuit pool.
        """
        jobs = []
        for i in range(n):
            name, circuit, shots, time = self.generate_circuit_qft_pool()
            jobs.append(job(name, circuit, shots, time + time_frame * step))
        return jobs

    def generate_job_qugan_pool(self, n, time_frame, step):
        """
        Generate n jobs from the QuGAN circuit pool.
        """
        jobs = []
        for i in range(n):
            name, circuit, shots, time = self.generate_circuit_qugan_pool()
            jobs.append(job(name, circuit, shots, time + time_frame * step))
        return jobs
    
    def generate_job_arith_pool(self, n, time_frame, step):
        """
        Generate n jobs from the arithmetic circuit pool.
        """
        jobs = []
        for i in range(n):
            name, circuit, shots, time = self.generate_circuit_arith_pool()
            jobs.append(job(name, circuit, shots, time + time_frame * step))
        return jobs

    def generate_job_sequence(self, i, path):
        """
        Generate a job from a specific sequence index in a directory.
        Returns a list containing a single job.
        """
        circuit_queue = []
        all_qasm_files = self.get_qasm_files(path)

        circuit = qasm.circuit_from_qasm(all_qasm_files[i])
        shots = random.uniform(50, 1000)
        time = random.uniform(0, 10)
        name = all_qasm_files[i].split('/')[-1]
        circuit_queue.append(job(name, circuit, 0, 0))
        return circuit_queue

    def generate_all_jobs(self):
        """
        Generate jobs from all circuits in the pool_2 directory.
        """
        circuit_queue = []
        all_qasm_files = self.get_qasm_files("/Users/mac/Desktop/qCloud/circuit/pool_2")
        for i in range(len(all_qasm_files)):
            circuit = qasm.circuit_from_qasm(all_qasm_files[i])
            shots = random.uniform(50, 1000)
            time = random.uniform(0, 10)
            name = all_qasm_files[i].split('/')[-1]
            circuit_queue.append(job(name, circuit, 0, 0))
        return circuit_queue

    def generate_single_job(self, time_frame, step, probability):
        """
        Generate a single job with the given parameters.
        Used by the parallel job generation method.
        """
        circuit, shots, time = self.generate_circuit(probability)
        return job(f"job_{time}", circuit, shots, time + time_frame * step)

    def generate_job_new(self, n, time_frame, step, probability):
        """
        Generate n jobs in parallel using multiprocessing for better performance.
        
        Args:
            n: Number of jobs to generate
            time_frame: Time frame for each step
            step: Current step number
            probability: Probability distribution for circuit sizes
            
        Returns:
            A list of job objects generated in parallel
        """
        # Create a tuple of arguments for each job
        args = [(time_frame, step, probability) for _ in range(n)]

        # Create a pool of processes and generate jobs in parallel
        with multiprocessing.Pool() as pool:
            jobs = pool.starmap(self.generate_single_job, args)

        return jobs


class job:
    """
    Represents a quantum job to be executed on the quantum cloud.
    Contains information about the circuit, required shots, and timing.
    """
    def __init__(self, name, circuit, shots, time):
        """
        Initialize a job with the given parameters.
        
        Args:
            name: Name/identifier of the job
            circuit: Quantum circuit to execute
            shots: Number of shots for the circuit execution
            time: Arrival time of the job
        """
        self.name = name
        self.circuit = circuit
        self.shots = shots
        self.time = time
        self.placement = None  # Will store the placement strategy when scheduled
        self.status = None     # Will track job status (e.g., queued, running, finished)
        global current_job_id
        self.id = current_job_id
        current_job_id += 1


class job_queue:
    """
    Manages a queue of jobs sorted by arrival time.
    """
    def __init__(self, job_list):
        """
        Initialize the job queue with a list of jobs.
        Jobs are automatically sorted by arrival time (earliest first).
        
        Args:
            job_list: List of job objects to be queued
        """
        self.queue = job_list
        self.queue.sort(key=lambda x: x.time, reverse=False)


def main():
    """Main function for testing the job generator."""
    print(random.choice([1, 2, 3]))
    a = job_generator()
    a.generate_circuit([0.33, 0.33, 0.34])


if __name__ == "__main__":
    main()
