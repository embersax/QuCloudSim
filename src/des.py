"""
Discrete Event Simulator (DES) for quantum cloud simulation.

This module implements a discrete event simulator that models events in the quantum cloud,
such as job completion and new job generation. The simulator maintains an event queue
and processes events in chronological order.
"""
import heapq
from abc import ABC, abstractmethod

class Event(ABC):
    """
    Abstract base class for all events in the simulation.
    
    Events are processed in chronological order based on their time attribute.
    Each specific event type must implement the execute method to define its behavior.
    """
    def __init__(self, time):
        """
        Initialize an event with a specific time.
        
        Args:
            time: The simulation time when this event occurs
        """
        self.time = time
        
    @abstractmethod
    def execute(self, simulation):
        """
        Execute the event in the simulation.
        
        Args:
            simulation: The DES instance that is running the simulation
        """
        pass

    def __lt__(self, other):
        """
        Compare events based on their time for priority queue ordering.
        
        Args:
            other: Another event to compare with
            
        Returns:
            True if this event should occur before the other event
        """
        if isinstance(other, Event):
            return self.time < other.time
        return NotImplemented


class FinishedJob(Event):
    """
    Event representing the completion of a quantum job.
    
    When a job finishes, resources are freed, and the scheduler checks if any 
    waiting jobs can now be scheduled.
    """
    def __init__(self, time, job, qpu, log, placement):
        """
        Initialize a job completion event.
        
        Args:
            time: The time when the job completes
            job: The job that completed execution
            qpu: Dictionary mapping QPU IDs to the number of qubits used on each QPU
            log: Log information about the job execution
            placement: The placement strategy used for the job
        """
        super().__init__(time)
        self.time = time
        self.job = job
        self.qpu = qpu
        self.log = log
        self.placement = placement

    def execute(self, des):
        """
        Execute the job completion event.
        
        This method:
        1. Frees the resources used by the completed job on each QPU
        2. Adds any unscheduled jobs back to the main job queue
        3. Attempts to schedule waiting jobs
        
        Args:
            des: The DES instance running the simulation
        """
        # Free resources on each QPU used by this job
        for qpu_id in self.qpu.keys():
            qpu = des.cloud.network.nodes[qpu_id]['qpu']
            nqubits = self.qpu[qpu_id]
            qpu.free_qubits(nqubits, self.job)
            
        # Move any unscheduled jobs back to the main job queue
        des.scheduler.job_queue.extend(des.scheduler.unscheduled_job)
        des.scheduler.unscheduled_job = []
        
        # Attempt to schedule more jobs if any are waiting
        if des.scheduler.job_queue:
            des.scheduler.schedule()


class GeneratingJob(Event):
    """
    Event representing the arrival of a batch of new jobs.
    
    This event models periodic job arrival patterns where batches of jobs
    arrive at fixed time intervals.
    """
    def __init__(self, time, job_list):
        """
        Initialize a job generation event.
        
        Args:
            time: The time when the jobs arrive
            job_list: A list of job objects that are arriving
        """
        super().__init__(time)
        self.job_list = job_list

    def execute(self, des):
        """
        Execute the job generation event.
        
        This method:
        1. Adds the new jobs to the job queue
        2. Attempts to schedule the new jobs
        
        Args:
            des: The DES instance running the simulation
        """
        des.scheduler.job_queue.extend(self.job_list)
        des.scheduler.schedule_choice()


class DES:
    """
    Discrete Event Simulator for quantum cloud simulation.
    
    Maintains an event queue and processes events in chronological order.
    Integrates with the cloud model and job scheduler.
    """
    def __init__(self, cloud=None, scheduler=None, logger=None):
        """
        Initialize the discrete event simulator.
        
        Args:
            cloud: The quantum cloud model
            scheduler: The job scheduler
            logger: Logger for recording simulation data
        """
        self.current_time = 0
        self.event_queue = []
        self.cloud = cloud
        self.scheduler = scheduler
        
        if scheduler is not None:
            if not self.scheduler.job_queue:
                self.scheduler.job_queue = []
            self.scheduler.des = self
            
        self.unpushed_event = []
        self.finished_job = []
        self.logger = logger

    def schedule_event(self, event):
        """
        Add an event to the simulation queue.
        
        Args:
            event: The event to schedule
        """
        heapq.heappush(self.event_queue, event)

    def run(self):
        """
        Run the simulation until no more events are in the queue.
        
        Processes events in chronological order, updating the simulation time
        and executing each event's behavior.
        """
        while self.event_queue:
            # Get the next event with the earliest time
            event = heapq.heappop(self.event_queue)
            
            # Process finished job events specially to log their completion
            if isinstance(event, FinishedJob):
                self.finished_job.append(event)
                print(f"Job finished at time {event.time}: {event.log}")
                
                # Log job completion details if a logger is available
                if self.logger:
                    available_qubits = self.cloud.get_available_qubits()
                    job_completion_time = event.time - event.placement.start_time
                    self.logger.log(
                        event.job.name, 
                        log=event.log, 
                        time=event.time,
                        original_time=event.placement.dag_longest_path_length, 
                        qubits=available_qubits,
                        start_time=event.placement.start_time, 
                        jct=job_completion_time
                    )

            # Update current simulation time
            self.current_time = event.time
            
            # Execute the event's behavior
            event.execute(self)


# Alias for backward compatibility
generatingJob = GeneratingJob
