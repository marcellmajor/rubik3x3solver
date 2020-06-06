from solution import *
from cube import *
import random
import gc

chromosome_size = 100
min_mutation_rate = 3
mutation_rate = min_mutation_rate
my_solution = Solution(chromosome_size)
my_solution.generate_random_genes()
print("Initial chromosome: ", my_solution.get_chromosome())
my_cube = Cube(
    "WBWBRRRGR",
    ["RRRYYYBBW","BWGGBGGYB","YRYWWWWWG","BYGRGBYGY"],
    "OOOOOOOOO"
)
my_cube.test_cube()
my_cube.print_colored()
# solved = my_cube.count_solved_faces()
# print("Solved faces:",str(solved))
# print("Solved edges:",str(my_cube.count_solved_edges()))
# print("Rotate Right")
# _temp_cube = my_cube.clone()
# _temp_cube.rotate_right(2)
# _temp_cube.test_cube()
# _temp_cube.print_colored()
# print("Rotate Left")
# _temp_cube = my_cube.clone()
# _temp_cube.rotate_left(2)
# _temp_cube.test_cube()
# _temp_cube.print_colored()
# print("Rotate Up")
# _temp_cube = my_cube.clone()
# _temp_cube.rotate_up(2)
# _temp_cube.test_cube()
# _temp_cube.print_colored()
# print("Rotate Down")
# _temp_cube = my_cube.clone()
# _temp_cube.rotate_down(2)
# _temp_cube.test_cube()
# _temp_cube.print_colored()
# print("Rotate Clockwise")
# _temp_cube = my_cube.clone()
# _temp_cube.rotate_cloclkwise(1)
# _temp_cube.test_cube()
# _temp_cube.print_colored()
# print("Rotate Backwards")
# _temp_cube = my_cube.clone()
# _temp_cube.rotate_backwards(1)
# _temp_cube.test_cube()
# _temp_cube.print_colored()

_index, _fitness = my_solution.calculate_fitness(my_cube.clone(), my_solution.get_chromosome())
print("Initial fitness:",str(_fitness))

# looping for solution
_counter = 0
_best_index = _index

# calulating initial chromosome
# _length = 0
# _best_chromosome = ""
# _index = 0
# while _length < chromosome_size:
#     _length += 1
#     _max_fitness = 0
#     _best_gene = ""
#     for _direction in gene_directions:
#         for _rotation in range(1,max_rotation + 1):
#             _new_gene = "{}{}".format(_direction, _rotation)
#             _first_cube = my_cube.clone()
#             _first_solution = Solution(_length)
#             _index, _fitness = _first_solution.calculate_fitness(_first_cube, _best_chromosome + _new_gene, mode = Mode.CUMULATED)
#             #print(_new_gene, _fitness)
#             if _fitness >= _max_fitness:
#                 if _fitness >= _max_fitness:
#                     if random.randint(1,2) == 2:
#                         _best_gene = _new_gene
#                         _max_fitness = _fitness
#                 else:
#                     _best_gene = _new_gene
#                     _max_fitness = _fitness
#     assert( _best_gene != "" )
#     _best_chromosome += _best_gene
#     #print("{}.step - best_gene:{} - chromosome:{}".format(_length, _best_gene, _best_chromosome))
# my_solution.set_chromosome( _best_chromosome )
# _index, _fitness = my_solution.calculate_fitness(my_cube.clone(), my_solution.get_chromosome(), _index)


# _act_cube = my_cube.clone()
# _index, _fitness = my_solution.calculate_fitness( _act_cube, "L1", 0, mode = Mode.LAST )
# print("Fitness: ", _fitness)
# _act_cube = my_cube.clone()
# _index, _fitness = my_solution.calculate_fitness( _act_cube, "L1L1L1", 2, mode = Mode.LAST )
# print("Fitness: ", _fitness)
# _index, _fitness = my_solution.calculate_fitness( _act_cube, "L1L1L1L1", 2, mode = Mode.LAST )
# print("Fitness: ", _fitness)
# exit(0)

N = 4
print("\nSTART: calculating all possible combination of moves up to {}\n".format(N))
_move_combinations_set = set()
_length = 0
_counter = 0
while _length <= N:
    print("Deepcopy/list conversion...")
    if _length == 0:
        _previuos_combinations = list( [""] )
    else:
        _previuos_combinations = list( _move_combinations_set )
    print("{}Counter: {}; N={}; _length={}".format(">"*_length,_counter, N, _length))
    print("combinations:",len(_previuos_combinations))
    _added_count = 0
    for _act_combo in _previuos_combinations:
        for _direction in gene_directions:
            for _rotation in range( 1, max_rotation + 1 ):
                #gc.disable()
                _move_combinations_set.add(
                    _act_combo + "{}{}".format(_direction, _rotation)
                )
                #Sgc.enable()
                _counter += 1
                _added_count += 1
                if _counter % 10000000 == 0:
                    print("Counter: {}; N={}; _length={}".format(_counter, N, _length))
    print("_added_count:",_added_count)
    _length += 1
print("Counter:", _counter)
_move_combinations = list( _move_combinations_set )
#_move_combinations.pop(0)
_num_combinations = len(_move_combinations)
print("Total number of combinations found:", _num_combinations)

print("\nSTART: trying to find best possible solution using precalculated move combinations\n")
_length = 0
_counter = 0
_max_len = chromosome_size
_best_chromosome = ""
_max_fitness = 0
_act_color_index = 4
while _length < _max_len:
    _selected_chromosome = ""
    for _act_combo in _move_combinations:
        _act_cube = my_cube.clone()
        _act_chromosome = _best_chromosome + _act_combo
        _act_solution = Solution( int( len( _act_chromosome ) / 2 ) )
        _act_color = colors[_act_color_index]
        _index, _fitness = _act_solution.calculate_fitness( _act_cube, _act_chromosome, mode = Mode.LAST, color=_act_color )
        if _fitness >= _max_fitness:
            _selected_chromosome = _act_chromosome
            _max_fitness = _fitness
        if _counter % 1000 == 0:
            print("Progress: {:.2f}% max_fitness={};length={};best_chromosome={}".format(_counter/_num_combinations*100, _max_fitness, N, _selected_chromosome), end="\r")
        _counter += 1
    if _selected_chromosome == "":
        break
    _best_chromosome = my_solution.simplify_chromosome(_selected_chromosome)
    if _best_chromosome == "":
        break
    if _max_fitness >= 10000:
        _test_cube = my_cube.clone()
        _index, _fitness = my_solution.calculate_fitness(_test_cube, my_solution.get_chromosome(), _best_index, color=colors[_act_color_index])
        _act_color_index = _test_cube.get_almost_complete_color( color=colors[_act_color_index] )
        _index, _fitness = my_solution.calculate_fitness(_test_cube, my_solution.get_chromosome(), _best_index, color=colors[_act_color_index])
    _length = int( len(_best_chromosome) / 2 )
    print("\n\n_max_fitness={};length={};best_chromosome={}".format( _max_fitness, _length, _best_chromosome ))
    _temp_cube = my_cube.clone()
    _genes = _best_chromosome
    _best_index = _length - 1
    _act_solution.calculate_fitness(_temp_cube, _genes, _best_index, mode = Mode.LAST, color=colors[_act_color_index])

print("Counter:", _counter)
#exit(0)
if _best_chromosome != "" and _max_fitness > 0:
    print("\n\nEND\nmax_fitness={};length={};best_chromosome={}\n\n".format( _max_fitness, _length, _best_chromosome ))
    my_solution.set_chromosome( _best_chromosome )
# # lucky chromosome
my_solution.set_chromosome( "B2C2L3U3C3U1D3B3D2B1D3L2C2L2C2" )
# my_solution.set_chromosome( "B2L2C3C3R2C2R2B3C1U3B1C3L3R3C2L3C2L2C3" )
_index, _fitness = my_solution.calculate_fitness(my_cube.clone(), my_solution.get_chromosome(), _best_index, color=colors[_act_color_index])

print("\nSTART genetic algorithm loop\n")
_last_change_counter = 0
_color_changed_at = 0
#while _fitness != solved_fitness and _counter < 500000:
while (not ((_fitness > (solved_fitness - chromosome_size)) and (_best_index < 10)) ) and _counter < 50000000:
    _temp_cube = my_cube.clone()
    _new_cube = my_cube.clone()
    _genes = my_solution.mutate_genes(mutation_rate)
    _new_solution = None
    _new_solution = Solution(chromosome_size)
    _new_solution.generate_random_genes()
    whichRND = ''
    if _best_index > -1:
        #crossover
        _genes_new = _new_solution.get_chromosome()
        _genes_old = _genes
        _crossover_point = (_best_index + 1) * 2
        _rnd = random.randint(1,3)
        if _rnd == 1:
            _new_solution.set_chromosome( _genes_old[:_crossover_point] + _genes_new[_crossover_point:] )
            whichRND = 'addnewaftercrosoverpoint'
        elif _rnd == 2:
            _new_solution.set_chromosome( _genes_new[:_crossover_point] + _genes_old[_crossover_point:] )
            whichRND = 'addnewbeforecrosoverpoint'
        else:
             _new_solution.set_chromosome( _genes_new )
             whichRND = 'fullnew'

    _new_genes = _new_solution.get_chromosome()
    _new_index, _new_fitness = _new_solution.calculate_fitness(_new_cube,_new_genes, color=colors[_act_color_index])
    _max_index, _act_fitness = my_solution.calculate_fitness(_temp_cube, _genes, color=colors[_act_color_index])

    if _new_fitness > _act_fitness:
        _genes = _new_genes
        _max_index = _new_index
        _act_fitness = _new_fitness
        _temp_cube = _new_cube
        
        if _act_fitness > _fitness:
            print("\n\n{}. RND ({}) taken over {}".format(_counter, whichRND, _act_fitness))
            my_solution = copy.deepcopy(_new_solution)

    if _act_fitness > _fitness:
        _last_change_counter = 0
        mutation_rate = min_mutation_rate
        my_solution.set_chromosome( _genes )
        _fitness = _act_fitness
        _best_index = _max_index
        print("\n\n{}. chromosome changed; index:{}, new: {}".format(_counter, _best_index, _genes))
        _temp_cube = my_cube.clone()
        my_solution.calculate_fitness(_temp_cube, _genes, _best_index, color=colors[_act_color_index])
        #_temp_cube.print_colored()
        _chromosome = my_solution.get_chromosome()
        print("{}. round, fitness: {:+.2f}; index: {}; chromosome_size={}; chromosome={}".format(_counter, _fitness, _best_index, len(_chromosome) / 2, _chromosome[:72]), end="\r")
        _color_changed_at = _counter
    _counter += 1
    _last_change_counter += 1
    if _counter % 1000 == 0:
        if _last_change_counter> 10000 and mutation_rate < ( chromosome_size / 2 ):
            mutation_rate += 1
            _last_change_counter = 0
        _chromosome = my_solution.get_chromosome()
        print("{}. round, fitness: {:+.2f}; change_count: {}; color: {}; rate: {}; index: {}; chromosome[:60]={}".format(_counter, _fitness, _last_change_counter, colors[_act_color_index], mutation_rate, _best_index, _chromosome[:60]), end="\r")
    if _fitness >= 10000 and (_counter - _color_changed_at)> 500000:
        _test_cube = my_cube.clone()
        _index, _fitness = my_solution.calculate_fitness(_test_cube, my_solution.get_chromosome(), _best_index, color=colors[_act_color_index])
        _act_color_index = _test_cube.get_almost_complete_color( color=colors[_act_color_index] )
        _index, _fitness = my_solution.calculate_fitness(_test_cube, my_solution.get_chromosome(), _best_index, color=colors[_act_color_index])
        _color_changed_at = _counter
print("\n\nCOUNTER = {}".format(_counter))
print()
_temp_cube = my_cube.clone()
_i, _f = my_solution.calculate_fitness( _temp_cube, my_solution.get_chromosome(), _best_index, color=colors[_act_color_index] )
print("{}. step, fitness: {}".format(_i, _f))
print("Solution: {}".format(my_solution.get_chromosome()[:(_best_index+1)*2]))