"""Handles the assignment of the machines for the jobs

There will be a N x M matrix, where M is number of machines, N is total number of jobs.
This matrix has the pheremone levels that job i is assigned to machine k (i,k). The
heuristic value (visibility) will either be the inverse of the processing time, or something with setuptime, maybe dynamic?
"""

class ACOMachine:
    def __init__(self, problem):
        self.problem = problem