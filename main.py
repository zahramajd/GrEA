import numpy as np
from optproblems.dtlz import DTLZ2
import random

# evaluation
# dt = DTLZ2(4,14)
# fitness_assignment(p, grid)
'''
Grid setting: grid is objevtive
Fitness assignment: GR, GCD, GCPD
Mating selection: Tournament Selection ? how many
Variation: simulated binary crossover and polynomial mutation with both distribution indexes 20
Environmental selection: Findout best, GR adjustment 
'''
# params
num_variables = 4
num_objectives = 3
n = 100
number_evaluations = 1
div = 5
number_parents = 50
pc = 1
pm = 1/num_variables
eta_c = 20
eta_m = 20

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

        return np.floor((fk - lbk) / dk)

    f_x = np.empty((0, num_objectives))
    for i in range(p.shape[0]):
        f_x = np.append(f_x, np.array(eval_func(p[i])).reshape(1, -1),axis=0)

    grid = np.apply_along_axis(funk, axis=1, arr=f_x)

    return grid

def fitness_assignment():
    pass

def mating_selection(p, grid, number_parents, num_variables, eval_func):

    def tournament_selection(choosed, eval_func, grid_choosed, grid):
        if(dominance(choosed[0], choosed[1], eval_func) or grid_dominance(grid_choosed[0], grid_choosed[1])):
            return choosed[0]
        elif (dominance(choosed[1], choosed[0], eval_func) or grid_dominance(grid_choosed[1], grid_choosed[0])):
            return choosed[1]
        elif (gcd(grid, grid_choosed[0])<gcd(grid, grid_choosed[1])):
            return choosed[0]
        elif (gcd(grid, grid_choosed[1])<gcd(grid, grid_choosed[0])):
            return choosed[1]
        elif random.uniform(0, 1) < 0.5:
            return choosed[0]
        else: return choosed[1]

    parents = np.empty((0, num_variables))
    for i in range(number_parents):
        rand_indices = np.random.choice(p.shape[0], 2, replace=False)
        winner = tournament_selection(p[rand_indices, :], eval_func, grid[rand_indices], grid).reshape(1,-1)
        parents = np.append(parents, winner, axis=0)

    return parents

def dominance(ind1, ind2, eval_func):
    eval_ind1 = np.array(eval_func(ind1)).reshape(1, -1)
    eval_ind2 = np.array(eval_func(ind2)).reshape(1, -1)

    all_more_equal = True
    one_more = False
    for i in range(eval_ind1.shape[1]):
        if (not(eval_ind1[0, i] <= eval_ind2[0, i])):
            all_more_equal = False
        
        if (eval_ind1[0, i] < eval_ind2[0, i]):
            one_more = True

    return (all_more_equal and one_more)

def grid_dominance(grid_ind1, grid_ind2):
    all_more_equal = True
    one_more = False
    for i in range(grid_ind1.shape[0]):
        if (not(grid_ind1[i] <= grid_ind2[i])):
            all_more_equal = False
        
        if (grid_ind1[i] < grid_ind2[i]):
            one_more = True

    return (all_more_equal and one_more)

def gcd(grid, grid_ind):

    def gd(grid_ind, grid_other):
        gd = 0
        for k in range(grid_ind.shape[0]):
            gd += np.abs(grid_ind[k]-grid_other[k])

        return gd

    gcd = 0
    for j in range(grid.shape[0]):
        diff =  grid.shape[1]- gd(grid_ind, grid[j])
        if(diff >0):
            gcd += diff

    return gcd

def variation(parents, pc, pm, eta_c, eta_m):

    # crossover
    i = 0
    children = np.empty((0, parents.shape[1]))
    while(i < parents.shape[0]):
        children =np.append(children, cxSimulatedBinary(parents[i], parents[i+1], eta_c), axis=0)
        i +=2

    # mutation
    low = np.amin(children, axis=0)
    up = np.amax(children, axis=0)
    for j in range(children.shape[0]):
        children[j] = mutPolynomialBounded(children[j], eta_m, low, up, pm)

    return children

def cxSimulatedBinary(ind1, ind2, eta):

    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        rand = random.random()
        if rand <= 0.5:
            beta = 2. * rand
        else:
            beta = 1. / (2. * (1. - rand))
        beta **= 1. / (eta + 1.)
        ind1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
        ind2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))

    return ind1, ind2

def mutPolynomialBounded(individual, eta, low, up, indpb):

    size = len(individual)
    for i, xl, xu in zip(range(size), low, up):
        if random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (xu - xl)
            x = min(max(x, xl), xu)
            individual[i] = x
    return individual

def environmental_selection():
    pass


"""
    Grid-Based Evolutionary Algorithm
"""

dt = DTLZ2(num_objectives, num_variables)
eval_func = dt.objective_function
p = initialize(n, num_variables)
t = 0

while(not termination(number_evaluations, t)):
    grid = grid_setting(p, div, eval_func, num_objectives)
    p_prime = mating_selection(p, grid, number_parents, num_variables, eval_func)
    p_double_prime = variation(p_prime, pc, pm, eta_c, eta_m)
    print(p_double_prime.shape)
    # p = environmental_selection(p + p_double_prime)
    t += 1