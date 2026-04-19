from dataclasses import dataclass

@dataclass
class SolutionData:
    assignments : dict
    assignment_cost : float
    opening_cost : float
    total_cost : float
    num_facilities : int
    