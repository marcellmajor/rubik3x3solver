#!python
#cython: language_level=3

from solution import *
from cube import *
import random
import gc
import threading
import time
from multiprocessing import Process, Lock, Queue

class Processing(Enum):
    SINGLE_THREAD = 1
    MULTI_THREAD = 2
    MULTI_PROCESSING = 3
    MEET_IN_THE_MIDDLE = 4

def main():
    chromosome_size = 24
    min_mutation_rate = 3
    mutation_rate = min_mutation_rate
    my_solution = Solution(chromosome_size)
    my_solution.generate_random_genes()
    print("Initial chromosome: ", my_solution.get_chromosome())
    my_cube = Cube(
        "GBRWORWOY",
        ["GOORYWYRR","YBBOGGWGG","OYROWBOGB","GWWYBBYGO"],
        "WRRWRYBYB"
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

    N = 6
    JUST_READING = False
    print("\nSTART: calculating all possible combination of moves up to {}\n".format(N))
    if not JUST_READING:
        _move_combinations_set = set()
        _length = 0
        _counter = 0
        while _length < N:
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
            gc.collect(2)
        print("Counter:", _counter)
        print("len(_move_combinations_set):", len(_move_combinations_set))
        print("GC")
        gc.collect(2)
        print("WRITING")
        fp = open("combinations.txt", "w")
        for _line in _move_combinations_set:
            fp.write(_line + "\n")
        fp.close()
        _move_combinations_set = {}
        _previuos_combinations = {}
        print("GC")
        gc.collect(2)
    #_move_combinations = list( _move_combinations_set )
    print("READING")
    fp = open('combinations.txt', 'r')
    _move_combinations = []
    line = fp.readline()
    if len(line) > 0:
        _move_combinations.append(line.strip())
    cnt = 1
    while line:
        line = fp.readline()
        if len(line) > 0:
            _move_combinations.append(line.strip())
        cnt += 1
        if cnt % 1000000 == 0:
            print("Read count = ", cnt, end="\r")
    fp.close()
    print("READ {} combinations".format(len(_move_combinations)))
    # _move_combinations.extend(
    #     [
    #         "U1L1U3L1U1L2U3L1",
    #         "U1C1U3C1U1C2U3C1",
    #         "U1R1U3R1U1R1U3R1",
    #         "R1D3R3D3R1D3R3D3",
    #         "D1L1D3L1D1L1D3L1",
    #         "L1U1L3U1L1U1L3U1",
    #         "U1B1U3B1U1B2U3R1"
    #     ]
    # )
    #_move_combinations.pop(0)
    _num_combinations = len(_move_combinations)
    print("Total number of combinations found:", _num_combinations)

    _best_chromosome = ""
    _act_color_index = 4
    _Processing_Mode = Processing.MEET_IN_THE_MIDDLE
    if _Processing_Mode == Processing.MULTI_THREAD:
        class myThread (threading.Thread):
            def __init__(self, combinations, start_index, end_index, thread_index, cube):
                threading.Thread.__init__(self)
                self.combinations = combinations
                self.start_index = start_index
                self.end_index = end_index
                self.thread_index = thread_index
                self.cube = cube
            
            def run(self):
                combinations = self.combinations
                start_index = self.start_index
                end_index = self.end_index
                thread_index = self.thread_index
                cube = self.cube
                print("Thread index: {}; start index: {}; end index:{}".format(thread_index, start_index, end_index) )
                best_fitness = 0
                selected_chromosome = ""
                for _index in range(start_index, end_index):
                    _act_combo = combinations[_index]
                    _act_cube = cube.clone()
                    _act_chromosome = _act_combo
                    _act_solution = Solution( int( len( _act_chromosome ) / 2 ) )
                    _act_index, _fitness = _act_solution.calculate_fitness( _act_cube, _act_chromosome, mode = Mode.LAST )
                    if _fitness >= best_fitness:
                        selected_chromosome = _act_chromosome
                        best_fitness = _fitness
                    if _index % 100000 == 0:
                        print("Thread index: {}; index: {:.2f}%".format(thread_index, (_index - start_index)/(end_index - start_index) * 100) )
                print("Thread index: {} completed".format(thread_index))
                return selected_chromosome

        print("\nSTART: trying to find best possible solution MULTITHREADED using precalculated move combinations\n")
        thread_number = 8
        _threads = []
        start = time.time()
        for i in range(0, thread_number):
            _start_index = _num_combinations * i // thread_number
            _end_index = _num_combinations * (i + 1) // thread_number
            _threads.append( myThread( _move_combinations, _start_index, _end_index, i, my_cube ) )
        for _thread in _threads:
            _thread.start()
        for _thread in _threads:
            #thread_best_chromosome = _thread.join()
            _thread.join()
            #print("Thread finished; best chromosome{}".format(thread_best_chromosome))
        end = time.time()
        print("Time elapsed:", end - start )

    elif _Processing_Mode == Processing.MULTI_PROCESSING:
        #multiprocessing implementation
        def f(l, i, q, p_list, cube):
            _workload_size = len(p_list)
            best_fitness = 0
            selected_chromosome = ""
            _act_solution = Solution( 2 )
            _act_cube = cube.clone()
            _top_face = cube.top_face
            _row_of_faces = cube.row_of_faces
            _bottom_face = cube.bottom_face
            l.acquire()
            try:
                print('Process started, index = {}, workload_size = {},'.format( i, _workload_size ) )
            finally:
                l.release()
            ## Processing
            for _index in range( 0, _workload_size ):
                _act_cube.set_faces(_top_face, _row_of_faces, _bottom_face)
                _act_solution.set_number_of_genes( int( len( p_list[_index] ) / 2 ) )
                _act_index, _fitness = _act_solution.calculate_fitness( _act_cube, p_list[_index], mode = Mode.LAST )
                if _fitness >= best_fitness:
                    selected_chromosome = p_list[_index]
                    best_fitness = _fitness
                if _index % 500000 == 0:
                    l.acquire()
                    try:
                        _percent = 100 * _index / _workload_size
                        print( 'Process {}. at {:.2f}%'.format(i, _percent) )
                        #gc.collect(1)
                    finally:
                        l.release()
            l.acquire()
            try:
                print('Process finished, index = ', i)
                q.put((selected_chromosome, best_fitness))
            finally:
                l.release()

        print("\nSTART: finding best possible solution multiprocessing/precalculated\n")
        thread_number = 6
        _lock = Lock()
        _queue = Queue()
        _processes = []
        print(">>CREATING PROCESSES")
        for num in range(thread_number - 1):
            if num == thread_number - 1 - 1:
                print("LAST 2")
                _start_index = _num_combinations * num // (thread_number - 1)
                _end_index = _num_combinations * (num + 1) // (thread_number - 1)
                _middle = _start_index + (_end_index - _start_index) // 2
                _processes.append(
                    Process(target=f, args=(_lock, num, _queue, _move_combinations[_start_index:_middle], my_cube))
                )
                _processes.append(
                    Process(target=f, args=(_lock, num + 1, _queue, _move_combinations[_middle + 1:_end_index], my_cube))
                )
            else:
                _start_index = _num_combinations * num // (thread_number - 1)
                _end_index = _num_combinations * (num + 1) // (thread_number - 1)
                _processes.append(
                    Process(target=f, args=(_lock, num, _queue, _move_combinations[_start_index:_end_index], my_cube))
                )
        _move_combinations = []
        gc.collect(2)
        print(">>STARTING PROCESSES")
        for num in range(thread_number):
            _processes[num].start()
        _is_alive = thread_number
        while _is_alive > 0:
            _is_alive = 0
            for num in range(thread_number):
                if _processes[num].is_alive():
                    _is_alive += 1
            print("_is_alive=", _is_alive, end="\n")
            time.sleep(10)
            #gc.collect(2)
        _max_fitness = 0
        for num in range(thread_number):
            _chromosome, _fitness = _queue.get()
            if _fitness > _max_fitness:
                _max_fitness = _fitness
                _best_chromosome = my_solution.simplify_chromosome( _chromosome )
            print( "{} {}".format( _chromosome, _fitness ) )
        for num in range(thread_number):
            _processes[num] = None
        gc.collect(2)
        _length = int( len(_best_chromosome) / 2 )
        _best_index = _length - 1
        _genes = _best_chromosome
        _temp_cube = my_cube.clone()
        my_solution.calculate_fitness(_temp_cube, _genes, _best_index, mode = Mode.LAST)

    elif _Processing_Mode == Processing.MEET_IN_THE_MIDDLE:
        # saving combos from solved cube
        if not JUST_READING: #only when there is no file
            solved_cube = Cube(
                "OOOOOOOOO",
                ["YYYYYYYYY","GGGGGGGGG","WWWWWWWWW","BBBBBBBBB"],
                "RRRRRRRRR"
            )
            _counter = 0
            print("WRITING")
            fp = open("combinations_solved.txt", "w")
            for _act_combo in _move_combinations:
                _act_cube = solved_cube.clone()
                _act_solution = Solution( int( len( _act_combo ) / 2 ) )
                _act_solution.calculate_fitness( _act_cube, _act_combo, mode = Mode.LAST, stop_when_solved = False )
                if _counter % 10000 == 0:
                    print("Progress: {:.2f}%".format(_counter/_num_combinations*100), end="\r")
                _counter += 1
                fp.write( _act_combo +":"
                    + _act_cube.top_face
                    + _act_cube.row_of_faces[0]
                    + _act_cube.row_of_faces[1]
                    + _act_cube.row_of_faces[2]
                    + _act_cube.row_of_faces[3]
                    + _act_cube.bottom_face + "\n" )
            fp.close()
        gc.collect(2)
        # reading combos from solved cube
        print("\nREADING")
        _counter = 0
        _max_fitness = 0
        fp = open('combinations_solved.txt', 'r')
        _solved_combinations = {}#set([])
        line = fp.readline()
        if len(line) > 0:
            _moves, _act_cube_string = line.strip().split(":")
            # _solved_combinations.add(_act_cube_string)
            if not _act_cube_string in _solved_combinations:
                _solved_combinations[_act_cube_string] = _moves
            elif len(_solved_combinations[_act_cube_string]) > len(_moves):
                _solved_combinations[_act_cube_string] = _moves
        _counter += 1
        while line:
            line = fp.readline()
            if len(line) > 0:
                _moves, _act_cube_string = line.strip().split(":")
                # _solved_combinations.add(_act_cube_string)
                if not _act_cube_string in _solved_combinations:
                    _solved_combinations[_act_cube_string] = _moves
                elif len(_solved_combinations[_act_cube_string]) > len(_moves):
                    _solved_combinations[_act_cube_string] = _moves
            _counter += 1
            if _counter % 10000 == 0:
                print("Progress: {:.2f}%".format(_counter/_num_combinations*100), end="\r")
        fp.close()
        print("READ {} combinations of solved cube".format(len(_solved_combinations)))
        gc.collect(2)
        # calculating combos from actual cube and matching with loaded ones
        print("MATCHING")
        _counter = 0
        for _act_combo in _move_combinations:
            _act_cube = my_cube.clone()
            _act_solution = Solution( int( len( _act_combo ) / 2 ) )
            _act_solution.calculate_fitness( _act_cube, _act_combo, mode = Mode.LAST, stop_when_solved = False )
            _act_cube_string = _act_cube.top_face \
                + _act_cube.row_of_faces[0] \
                + _act_cube.row_of_faces[1] \
                + _act_cube.row_of_faces[2] \
                + _act_cube.row_of_faces[3] \
                + _act_cube.bottom_face
            if _act_cube_string in _solved_combinations:
                print("MATCH FOUND!!!")
                _to_reverse = _solved_combinations[_act_cube_string]
                print(_to_reverse)
                f = lambda s,n:s and f(s[n:],n)+s[:n]
                __reversed = f(_to_reverse, 2)
                _reversed = ""
                for i in range(len(__reversed)):
                    if __reversed[i] == '1':
                        _reversed += '3'
                    elif __reversed[i] == '3':
                        _reversed += '1'
                    else:
                        _reversed += __reversed[i]
                print(_act_combo + "<=>" + _reversed)
                _best_chromosome = _act_combo + _reversed
                _max_fitness = solved_fitness
                _length = int( len(_best_chromosome) / 2 )
                _best_index = _max_fitness - 1
                _test_cube = my_cube.clone()
                _index, _fitness = my_solution.calculate_fitness(_test_cube,_best_chromosome, _best_index, color=colors[_act_color_index])
                exit(0)
                break
            if _counter % 10000 == 0:
                print("Progress: {:.2f}%".format(_counter/_num_combinations*100), end="\r")
            _counter += 1

    elif _Processing_Mode == Processing.SINGLE_THREAD:
        print("\nSTART: trying to find best possible solution using precalculated move combinations\n")
        _length = 0
        _counter = 0
        _max_len = chromosome_size
        _max_fitness = 0
        while _length < _max_len:
            start = time.time()
            _selected_chromosome = ""
            for _act_combo in _move_combinations:
                _act_cube = my_cube.clone()
                _act_chromosome = _best_chromosome + _act_combo
                _act_solution = Solution( int( len( _act_chromosome ) / 2 ) )
                _act_color = colors[_act_color_index]
                _index, _fitness = _act_solution.calculate_fitness( _act_cube, _act_chromosome, mode = Mode.LAST, color=_act_color )
                if _fitness > _max_fitness:
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
            end = time.time()
            print("Time elapsed:", end - start )
            print("Counter:", _counter)

    #exit(0)
    # _best_chromosome = "L3U2C2L1B1U3B2U3R2U2B3U2D2L2C2L1C2R1U1L3"
    # my_solution.set_chromosome( _best_chromosome )
    # _best_index = 19
    #_index, _fitness = my_solution.calculate_fitness(my_cube.clone(), my_solution.get_chromosome(), _best_index, color=colors[_act_color_index])
    if _best_chromosome != "" and _max_fitness > 0:
        print("\n\nEND\nmax_fitness={};length={};best_chromosome={}\n\n".format( _max_fitness, _length, _best_chromosome ))
        my_solution.set_chromosome( _best_chromosome )
    # # lucky chromosome
    # my_solution.set_chromosome( "B2C2L3U3C3U1D3B3D2B1D3L2C2L2C2" )
    # my_solution.set_chromosome( "B2L2C3C3R2C2R2B3C1U3B1C3L3R3C2L3C2L2C3" )

    # actual solution
    #my_solution.set_chromosome( "L3U2C2L1B1U3B2U3R2U2B3U2D2L2C2L1C2R1U2L3" )

    print("\nSTART genetic algorithm loop\n")
    _last_change_counter = 0
    #while _fitness != solved_fitness and _counter < 500000:
    while (not ((_fitness > (solved_fitness - chromosome_size)) and (_best_index < 25)) ) and _counter < 1000000000:
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
            my_solution.set_chromosome( _genes, simplify = True )
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
            if _last_change_counter> 50000 and mutation_rate < ( chromosome_size / 2 ):
                mutation_rate += 1
                _last_change_counter = 0
            _chromosome = my_solution.get_chromosome()
            print("{}. round, fitness: {:+.2f}; change_count: {}; color: {}; rate: {}; index: {}; chromosome[:60]={}".format(_counter, _fitness, _last_change_counter, colors[_act_color_index], mutation_rate, _best_index, _chromosome[:60]), end="\r")
    print("\n\nCOUNTER = {}".format(_counter))
    print()
    _temp_cube = my_cube.clone()
    _i, _f = my_solution.calculate_fitness( _temp_cube, my_solution.get_chromosome(), _best_index, color=colors[_act_color_index] )
    print("{}. step, fitness: {}".format(_i, _f))
    print("Solution: {}".format(my_solution.get_chromosome()[:(_best_index+1)*2]))

if __name__ == "__main__":
    main()