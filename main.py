import numpy as np
from optproblems.dtlz import DTLZ2

# class of individual
# evaluation
# i = Individual()
# dt = DTLZ2(4,14)

'''
Grid setting: grid is objevtive
Fitness assignment: GR, GCD, GCPD
Mating selection: Tournament Selection
Variation: ??
Environmental selection: Findout best, GR adjustment 
'''

# params
num_variables = 4
num_objectives = 3
n = 100
number_evaluations = 1
div = 5

def initialize(n, num_variables):
    return np.random.rand(n, num_variables)

def termination(number_evaluations, t):
    if t <= number_evaluations:
        return False
    else: return True

def grid_setting(p, div, eval_func, num_objectives):

    def funk(fk):
        mink = np.amin(fk)
        maxk = np.amax(fk)

        lbk = mink - (maxk - mink)/ (2 * div)
        ubk = maxk - (maxk - mink)/ (2 * div)
        dk = (ubk - lbk) / div

        return (fk - lbk) / dk


    f_x = np.empty((0, num_objectives))
    for i in range(p.shape[0]):
        f_x = np.append(f_x, np.array(eval_func(p[i])).reshape(1, -1),axis=0)

    grid = np.apply_along_axis(funk, axis=1, arr=f_x)

    return grid

def fitness_assignment():
    pass

def mating_selection():
    pass

def variation():
    pass

def environmental_selection():
    pass



"""
    Grid-Based Evolutionary Algorithm
"""

dt = DTLZ2(num_objectives, num_variables)

p = initialize(n, num_variables)
t = 0
while(not termination(number_evaluations, t)):
    grid = grid_setting(p, div, dt.objective_function, num_objectives)
    fitness_assignment(p, grid)
    # p_prime = mating_selection(p)
    # p_double_prime = variation(p_prime)
    # p = environmental_selection(p + p_double_prime)
    t += 1