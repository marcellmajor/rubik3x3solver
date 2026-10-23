# use deap for GA implementation
from deap import base, creator, tools, algorithms
import random
# use the cube module for cube representation and manipulation
from cube import *
import numpy as np
import matplotlib.pyplot as plt

# initialize random seed, from time in ticks
random.seed()

# genetic algorithm parameters
POPULATION_SIZE = 5000
GENERATIONS = 5000
MUTATION_PROB = 0.9
CROSSOVER_PROB = 0.001
HALL_OF_FAME_SIZE = 1
MAX_GENES = 12 # maximum number of genes in a chromosome (26)

min_moves = 0
best_fitness_ever = 1000.0

# creae mapping for moves defined in solution.py to integers for gene representation
# each gene will represent only one move in a direction, with a number of quarter turns
# genes = [
#     'R0', 'R1', 'R2', 'R3',
#     'L0', 'L1', 'L2', 'L3',
#     'U0', 'U1', 'U2', 'U3',
#     'D0', 'D1', 'D2', 'D3',
#     'C0', 'C1', 'C2', 'C3',
#     'B0', 'B1', 'B2', 'B3',
# ]

genes = [
    'R1', 'R2', 'R3',
    'L1', 'L2', 'L3',
    'U1', 'U2', 'U3',
    'D1', 'D2', 'D3',
    'C1', 'C2', 'C3',
    'B1', 'B2', 'B3',
]

# genes = [
#     'R1', 'R2',
#     'L1', 'L2',
#     'U1', 'U2',
#     'D1', 'D2',
#     'C1', 'C2',
#     'B1', 'B2',
# ]

# genes = [
#     'R1',
#     'L1',
#     'U1',
#     'D1',
#     'C1',
#     'B1',
# ]

gene_directions = ['R', 'L', 'U', 'D', 'C', 'B']

def print_best( best_individual ):
    print("Best individual moves up to min_moves ({}):".format(min_moves))
    solution_moves = [genes[gene] for gene in best_individual[:min_moves]]
    print("Solution moves:", solution_moves)
    # create a clone of the original cube and apply the solution moves to verify
    min_cube = my_cube.clone()
    for gene in best_individual[:min_moves]:
        move = genes[gene]
        rotate(min_cube, move)
    print("Best cube at best position (after {} moves):".format(min_moves))
    min_cube.print_as_python_list()
    min_cube.print_colored()

# a chromosome will be a list of integers, each representing a gene from the above list
GENE_COUNT = len(genes)
# MAX_ROTATION = 3  # maximum number of quarter turns per move

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        
        # Every 100 generations, prompt to print best
        if gen % 25 == 0 and gen > 0:
            best_so_far = halloffame.items[0] if halloffame.items else None
            if best_so_far:
                best_fitness = best_so_far.fitness.values[0]
                solution_moves = [genes[gene] for gene in best_so_far]
                print(f">>> BEST SO FAR (Gen {gen}): Fitness = {best_fitness}, Moves = {solution_moves}")
                print("min_moves = ", min_moves )
                print_best( best_so_far )

    return population, logbook

# inititialize the cube to be solved, this is a scrambled cube

my_cube = Cube(
                             "OGGRGGGOG",
    ["RWWRRORRG","BBRBWGOWB","WGOROOROO","YYYYYYYYY"],
                             "WWBBBBWWB"
)

# Best individual moves up to min_moves (18):
# Solution moves: ['D1', 'C1', 'D1', 'C3', 'C3', 'R3', 'D3', 'R1', 'D1', 'C1', 'U3', 'L1', 'U1', 'R1', 'U3', 'L3', 'U1', 'R3']
# Best cube at best position (after 18 moves):
my_cube = Cube(
                              "GGGBGGGOG",
[ "RWRRROYRR",  "WWWWWRWRW",  "OGOGOOOOY",  "YYYYYYRYO", ],
                              "BBBBBBBWB"
)

# Best individual moves up to min_moves (18):
# Solution moves: ['R1', 'R1', 'C3', 'R3', 'C3', 'R1', 'U1', 'R3', 'U3', 'R3', 'C1', 'R3', 'U1', 'R3', 'U3', 'R3', 'C1', 'R2']
# Best cube at best position (after 18 moves):
# my_cube = Cube(
#                               "GGGBGGGOG",
# [ "RWRRRORRR",  "WWWWWRWRW",  "OGOGOOOYO",  "YYYYYYYOY", ],
#                               "BBBBBBBWB"
# )

# Best individual moves up to min_moves (14):
# Solution moves: ['R3', 'R1', 'U2', 'D1', 'R1', 'C2', 'D2', 'B2', 'L1', 'C2', 'U2', 'B2', 'D3', 'U2']
# Best cube at best position (after 14 moves):
# my_cube = Cube(
#                               "GGGBGGGOG",
# [ "RWRRRGRRR",  "WWWRWWWRW",  "OGOOOOOOO",  "YYYYYYYYY", ],
#                               "BBBBBBBWB"
# )

# Best individual moves up to min_moves (12):
# Solution moves: ['C2', 'U1', 'U1', 'U3', 'U1', 'B2', 'L3', 'B2', 'U1', 'U1', 'C2', 'R3']
# Best cube at best position (after 12 moves):
my_cube = Cube(
                              "GGGWGGGOG",
[ "RWRRRGRRR",  "WRWRWWWWW",  "OGOOOOOOO",  "YYYYYYYYY", ],
                              "BBBBBBBBB"
)

# the solved cube is:
# solved_cube = Cube(
#                              "GGGGGGGGG",
#     ["RRRRRRRRR","WWWWWWWWW","OOOOOOOOO","YYYYYYYYY"],
#                              "BBBBBBBBB"
# )
my_cube.test_cube()
my_cube.print_colored()

# create individual, using DEAP's creator module, chromosome is a list of integers,
# mapped to items in the genes list
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# define gene as random integer from 0 to GENE_COUNT-1
toolbox.register("gene", random.randint, 0, GENE_COUNT - 1)
# define individual as a list of genes, length MAX_GENES
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=MAX_GENES)
# define population as a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# define mutation: randomly change one gene in the chromosome
def mutate_individual(individual):
    gene_index = random.randint( 0, MAX_GENES - 1)
    individual[gene_index] = random.randint(0, GENE_COUNT - 1)
    return individual,

# define an alternative mutation where the gene index is likely to be a higher index
def mutate_individual_biased(individual):
    gene_index = int(random.triangular(0, MAX_GENES - 1, MAX_GENES - 1))
    individual[gene_index] = random.randint(0, GENE_COUNT - 1)
    return individual,

translated_algs = [
    # Beginner LBL
    'R3D3R1D1',  # 1st Layer Corners: R' D' R D
    'U1R1U3R3U3C3U1C1',  # 2nd Layer Edges (right): U R U' R' U' F' U F
    'U3L3U1L1U1C1U3C3',  # 2nd Layer Edges (left): U' L' U L U F U' F'
    'C1R1U1R3U3C3',  # Yellow Cross: F R U R' U' F'
    'R1U1R3U1R1U2R3',  # Yellow Edges: R U R' U R U2 R'
    'U1R1U3L3U1R3U3L1',  # Yellow Corners (position): U R U' L' U R' U' L
    'R3D3R1D1',  # Yellow Corners (orient): R' D' R D (base for x2 or x4)
    'C1U1R1U3R3C3',  # Alt Yellow Cross: F U R U' R' F'

    # Common Triggers
    'R1U1R3U3',  # Sexy Move: R U R' U'
    'R3C1R1C3',  # Sledgehammer: R' F R F'
    'R1U3L3U1R3U3L1U1',  # Niklas: R U' L' U R' U' L U
    'U1R1U3R3',  # Reverse Sexy: U R U' R'

    # Advanced CFOP: OLL Examples
    'R1U1R3U1R1U2R3',  # Sune: R U R' U R U2 R'
    'R1U1R3U3R3C1R1C3',  # T-Shape: R U R' U' R' F R F'
    'R1U1R3U3R3C1R2U1R3U3C3',  # Fish: R U R' U' R' F R2 U R' U' F'
    'R1U2R3U3R1U1R3U3R1U3R3',  # Cross: R U2 R' U' R U R' U' R U' R'

    # Advanced CFOP: PLL Examples (standard moves only)
    'R1U1R3U3R3C1R2U3R3U3R1U1R3C3',  # T-Perm: R U R' U' R' F R2 U' R' U' R U R' F'
    'R1U1R3C3R1U1R3U3R3C1R2U3R3'   # Jb-Perm: R U R' F' R U R' U' R' F R2 U' R'
]

# define a mutation which uses the translated algorithms to replace a sequence of genes
def mutate_individual_with_alg(individual):
    alg = random.choice(translated_algs)
    alg_length = len(alg) // 2  # each move is represented by 2 characters
    if alg_length > MAX_GENES:
        return individual,
    start_index = random.randint(0, MAX_GENES - alg_length)
    for i in range(alg_length):
        move = alg[i*2:i*2+2]
        if move in genes:
            gene_index = genes.index(move)
            individual[start_index + i] = gene_index
    return individual,

# define mutation which is using the unbiased and biased versions half and half
def mutate_individual_mixed(individual):
    _random = random.random() # value between 0 and 1
    if _random < 0.45:
        return mutate_individual(individual)
    elif _random < 0.9:
        return mutate_individual_biased(individual)
    else:
        return mutate_individual_with_alg(individual)

# toolbox.register("mutate", mutate_individual)
toolbox.register("mutate", mutate_individual_mixed)

# define crossover: two point crossover
# toolbox.register("mate", tools.cxTwoPoint)
# define crossover: one point crossover
toolbox.register("mate", tools.cxOnePoint)

# rotate function, passed by reference to fitness function
def rotate(actual_cube, _act_gene):
    if _act_gene[0] == 'R':
        actual_cube.rotate_right(int(_act_gene[1]))
    if _act_gene[0] == 'L':
        actual_cube.rotate_left(int(_act_gene[1]))
    elif _act_gene[0] == 'U':
        actual_cube.rotate_up(int(_act_gene[1]))
    elif _act_gene[0] == 'D':
        actual_cube.rotate_down(int(_act_gene[1]))
    elif _act_gene[0] == 'C':
        actual_cube.rotate_cloclkwise(int(_act_gene[1]))
    elif _act_gene[0] == 'B':
        actual_cube.rotate_backwards(int(_act_gene[1]))
    else:
        assert( _act_gene[0] in gene_directions)

# count the misplaced stickers in a cube, each face should have the color of the middle sticker (5th out of the 9 faces)
def misplaced_stickers(cube):
    misplaced = 0
    faces = [
        cube.top_face, # green
        cube.row_of_faces[0], # red
        cube.row_of_faces[1], # white
        cube.row_of_faces[2], # orange
        cube.row_of_faces[3], # yellow
        cube.bottom_face # blue
    ]
    for face in faces:
        center_color = face[4]
        # if(face[0] != center_color):
        #     misplaced += 1
        # if(face[2] != center_color):
        #     misplaced += 1
        # if(face[6] != center_color):
        #     misplaced += 1
        # if(face[8] != center_color):
        #     misplaced += 1
        for sticker in face:
            if sticker != center_color:
                misplaced += 1
    return misplaced

# define fitness function
def evaluate_individual(individual):
    # create a copy of the cube to manipulate
    test_cube = my_cube.clone()
    # apply the moves represented by the individual's genes
    for gene in individual:
        move = genes[gene]
        rotate(test_cube, move)
    # calculate fitness as the number of misplaced stickers
    fitness = misplaced_stickers(test_cube)
    return (fitness,)

# print the fittness of the initial cube
initial_fitness = misplaced_stickers(my_cube)
print("Initial cube fitness (misplaced stickers):", initial_fitness)

# Manhattan distance heuristic
def manhattan_distance(cube):
    distance = 0
    faces = [
        cube.top_face, # green
        cube.row_of_faces[0], # red
        cube.row_of_faces[1], # white
        cube.row_of_faces[2], # orange
        cube.row_of_faces[3], # yellow
        cube.bottom_face # blue
    ]
    
    # Map each color to its proper offset in the solved cube layout
    color_to_offset = {
        'G': (0, 0),   # Green (top)
        'R': (0, 3),   # Red (front)
        'W': (0, 6),   # White (right)
        'O': (3, 0),   # Orange (left)
        'Y': (3, 3),   # Yellow (back)
        'B': (3, 6),   # Blue (bottom)
    }
    
    goal_positions = {
        'G': [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)],
        'R': [(0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)],
        'W': [(0,6), (0,7), (0,8), (1,6), (1,7), (1,8), (2,6), (2,7), (2,8)],
        'O': [(3,0), (3,1), (3,2), (4,0), (4,1), (4,2), (5,0), (5,1), (5,2)],
        'Y': [(3,3), (3,4), (3,5), (4,3), (4,4), (4,5), (5,3), (5,4), (5,5)],
        'B': [(3,6), (3,7), (3,8), (4,6), (4,7), (4,8), (5,6), (5,7), (5,8)],
    }
    
    for face in faces:
        center_color = face[4]
        goal_pos = goal_positions[center_color]
        offset_row, offset_col = color_to_offset[center_color]
        
        for idx, sticker in enumerate(face):
            if sticker != center_color:
                # Current position in global coordinates (using center color's coordinate system)
                local_row = idx // 3
                local_col = idx % 3
                current_row = offset_row + local_row
                current_col = offset_col + local_col
                
                # Goal position for this sticker color
                goal_pos_for_sticker = goal_positions[sticker]
                goal_row, goal_col = goal_pos_for_sticker[idx]
                
                # Manhattan distance
                distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    
    return distance

# define an alternative fittness function that uses Manhattan distance as heuristic
# which mmeans summing up the distances of each sticker from its goal position
def evaluate_individual_manhattan(individual):
    # create a copy of the cube to manipulate
    test_cube = my_cube.clone()
    # apply the moves represented by the individual's genes
    for gene in individual:
        move = genes[gene]
        rotate(test_cube, move)
    # calculate fitness as the sum of Manhattan distances of each sticker from its goal position
    fitness = manhattan_distance(test_cube)
    return (fitness,)

# same as fittness with manhttan distance but returns the smallest distance found during the moves
def evaluate_individual_manhattan_min(individual):
    # create a copy of the cube to manipulate
    test_cube = my_cube.clone()
    min_distance = manhattan_distance(test_cube)
    _num_moves = 0

    # apply the moves represented by the individual's genes
    for gene in individual:
        _num_moves += 1
        move = genes[gene]
        rotate(test_cube, move)
        distance = manhattan_distance(test_cube)
        if distance < min_distance:
            min_distance = distance
    # use the distance as a fractional value to prefer shorter solutions
    fitness = min_distance
    return (fitness,)

# fittnes function that uses the sum of MAnhattan distance and misplaced stickers
# and prefers solutions with fewer moves
weight_manhattan = 1.0
weight_misplaced = 0.0
def evaluate_individual_manhattan_misplaced(individual):
    global min_moves
    global best_fitness_ever
    # create a copy of the cube to manipulate
    test_cube = my_cube.clone()
    min_distance = manhattan_distance(test_cube)
    min_misplaced = misplaced_stickers(test_cube)
    _min_sum = weight_manhattan * min_distance + weight_misplaced * min_misplaced + ( min_moves / MAX_GENES )
    _num_moves = 0

    # apply the moves represented by the individual's genes
    for gene in individual:
        _num_moves += 1
        move = genes[gene]
        rotate(test_cube, move)
        distance = manhattan_distance(test_cube)
        misplaced = misplaced_stickers(test_cube)
        _sum = weight_manhattan * distance + weight_misplaced * misplaced + ( _num_moves / MAX_GENES )
        if _sum < _min_sum:
            if _sum < best_fitness_ever:
                min_moves = _num_moves
                best_fitness_ever = _sum
            _min_sum = _sum
    # use the distance and misplaced as a fractional value to prefer shorter solutions
    fitness = _min_sum
    return (fitness,)

# register the evaluation function
# toolbox.register("evaluate", evaluate_individual)
# toolbox.register("evaluate", evaluate_individual_manhattan)
# toolbox.register("evaluate", evaluate_individual_manhattan_min)
toolbox.register("evaluate", evaluate_individual_manhattan_misplaced)

# define tournament selection, using elite selection to keep best individuals
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("select_elite", tools.selBest)

# statistics to collect
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("med", np.median, axis=0)
stats.register("min", np.min, axis=0)
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

# print the Manhattan distance of the initial cube
initial_manhattan = manhattan_distance(my_cube)
print("Initial cube Manhattan distance:", initial_manhattan)

# run the genetic algorithm, use eaSimple algorithm from DEAP with elitism
population, logbook = eaSimpleWithElitism(
    population=toolbox.population(n=POPULATION_SIZE),
    toolbox=toolbox,
    cxpb=CROSSOVER_PROB,
    mutpb=MUTATION_PROB,
    ngen=GENERATIONS,
    stats=stats,
    halloffame=hof,
    verbose=True)

# use easimple as in DEAP
# population, logbook = algorithms.eaSimple(
#     population=toolbox.population(n=POPULATION_SIZE),
#     toolbox=toolbox,
#     cxpb=CROSSOVER_PROB,
#     mutpb=MUTATION_PROB,
#     ngen=GENERATIONS,
#     stats=stats,
#     halloffame=hof,
#     verbose=True)

# print the best solution found
best_individual = hof.items[0]
print("Best individual is:", best_individual)
print("with fitness:", best_individual.fitness.values[0])
# print the sequence of moves
solution_moves = [genes[gene] for gene in best_individual]
print("Solution moves:", solution_moves)

# create a clone of the original cube and apply the solution moves to verify
solved_cube = my_cube.clone()
for move in solution_moves:
    rotate(solved_cube, move)
print("Cube after applying solution moves:")
solved_cube.print_colored()
# print the fitness of the solved cube
final_fitness = misplaced_stickers(solved_cube)
print("Final cube fitness (misplaced stickers):", final_fitness)

print_best( best_individual )

# plot the progression of the genetic algorithm
minvalues, avgvalues = logbook.select( "min", "avg" )
plt.plot([i[0] for i in minvalues], color='r', label='fitness min over genertions')
plt.plot([i[0] for i in avgvalues], color='b', label='fitness avg over genertions')

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("GA progression")
plt.legend()
plt.show()
