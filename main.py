import numpy as np
from optproblems.dtlz import DTLZ2
import random

# evaluation
# dt = DTLZ2(4,14)
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
n = 10
number_evaluations = 1
div = 5
number_parents = 20
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
    gcd = 0
    for j in range(grid.shape[0]):
        diff =  grid.shape[1]- gd(grid_ind, grid[j])
        if(diff >0):
            gcd += diff
    gcd = gcd - grid.shape[1]
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

def environmental_selection(p, p_double_prime, div, eval_func, num_objectives):
    
    P = np.append(p, p_double_prime, axis=0)
    N = p.shape[0]
    Q = np.empty((0, p.shape[1]))
    F = pareto_dominance_sort(P, eval_func)
    index_critical = find_index_critical(F, N)

    for i in range(index_critical):
        Q = np.append(Q, F[i], axis=0)

    if(Q.shape[0]==N):
        return Q

    grid_f = grid_setting(F[index_critical], div, eval_func, num_objectives)
    gr_f, gcd_f, gcdp_f= env_initialization(F[index_critical], grid_f, eval_func)
    while(Q.shape[0]< N):
        q, q_index = findout_best(F[index_critical], gr_f, gcd_f, gcdp_f)
        Q = np.append(Q, q.reshape(1,-1), axis=0)
        F[index_critical] = np.delete(F[index_critical], q_index, axis=0)
        gcd_f = gcd_calculation(grid_f, grid_f[q_index], gcd_f)
        gr_f = gr_adjustment(F[index_critical], q, q_index, grid_f, gr_f, num_objectives)
    return Q

def pareto_dominance_sort(p, eval_func):
    F = []
    while(p.shape[0] > 0):
        front = np.empty((0, p.shape[1]))
        front_indcies = []
        for i in range(p.shape[0]):
            non_dom = True
            for j in range(p.shape[0]):
                if(not i==j):
                    if(dominance(p[j], p[i], eval_func)):
                        non_dom = False
            
            if (non_dom):
                front = np.append(front, p[i].reshape(1,-1), axis=0)
                front_indcies.append(i)

        p = np.delete(p, front_indcies, axis=0)
        F.append(front)

    return F

def find_index_critical(F, N):
    current_size = 0
    i = 0
    while( i < len(F)):
        if( (current_size + F[i].shape[0]) <= N):
            current_size += F[i].shape[0]
            i += 1
        else:
            break
    return i

def env_initialization(p, grid, eval_func):
    gr = gr_assignment(grid)
    gcdp = gcpd_assignment(p, eval_func)
    gcd = np.zeros((p.shape[0], 1))
    return gr, gcd, gcdp

def gr_assignment(grid):
    gr = np.empty((0,1))
    for i in range(grid.shape[0]):
        gr = np.append(gr, [[np.sum(grid[i])]], axis=0)
    return gr

def gcpd_assignment(p, eval_func):

    def funk(fk):
        mink = np.amin(fk)
        maxk = np.amax(fk)
        lbk = mink - (maxk - mink)/ (2 * div)
        ubk = maxk - (maxk - mink)/ (2 * div)
        dk = (ubk - lbk) / div
        gkx = np.floor((fk - lbk) / dk)
        return np.power((fk - (lbk + gkx * dk) / dk), 2)

    f_x = np.empty((0, num_objectives))
    for i in range(p.shape[0]):
        f_x = np.append(f_x, np.array(eval_func(p[i])).reshape(1, -1),axis=0)

    grid_tmp = np.apply_along_axis(funk, axis=1, arr=f_x)
    gcpd = np.sqrt(np.sum(grid_tmp, axis=1).reshape(-1,1))
    return gcpd 

def findout_best(p, gr, gcd, gcpd):
    q_index = 0
    for i in range(p.shape[0]):
        if(gr[i]<gr[q_index]):
            q_index = i
        elif(gr[i]==gr[q_index]):
            if(gcd[i]<gcd[q_index]):
                q_index = i
            elif(gcd[i]==gcd[q_index]):
                if(gcpd[i]<gcpd[q_index]):
                    q_index = i        
    return p[q_index], q_index

def gr_adjustment(P, q, q_index, grid, gr, num_objectives):
    m = num_objectives
    # same coordinate
    for i in range(P.shape[0]):
        if (not i==q_index):
            if((grid[q_index] == grid[i]).all()):
                gr[i] += (m+2)

    # grid dominated
    for i in range(P.shape[0]):
        if (not i==q_index):
            if(grid_dominance(grid[q_index], grid[i])):
                gr[i] += m

    # grid dominated by neighbours
    pd = np.zeros((P.shape[0], 1))
    for i in range(pd.shape[0]):
        neighbour = (m > gd(grid[q_index], grid[i]))
        if(neighbour and not grid_dominance(grid[q_index], grid[i]) and not ((grid[q_index] == grid[i]).all())):
            if(pd[i] < (m - gd(grid[i], grid[q_index]))):
                pd[i] = m - gd(grid[i], grid[q_index])
                for r in range(pd.shape[0]):
                    if(not (r==i) and grid_dominance(grid[i], grid[r]) and not (grid_dominance(grid[q_index], grid[r]) or grid[q_index] == grid[r]).all()):
                        if(pd[r]<pd[i]):
                            pd[r] = pd[p]

    for i in range(pd.shape[0]):
        if(not (grid_dominance(grid[q_index], grid[i])) or (grid[q_index] == grid[i]).all()):
            gr[i] = np.add(gr[i], pd[i])

    return gr

def gd(grid_ind, grid_other):
    gd = 0
    for k in range(grid_ind.shape[0]):
        gd += np.abs(grid_ind[k]-grid_other[k])
    return gd

def gcd_calculation(grid, q_grid, gcd):

    for j in range(grid.shape[0]):
        diff =  grid.shape[1]- gd(q_grid, grid[j])
        if(diff >0):
            gcd[j] += diff

    return gcd

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
    p = environmental_selection(p, p_double_prime, div, eval_func, num_objectives)
    t += 1