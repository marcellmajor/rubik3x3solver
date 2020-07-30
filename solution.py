#!python
#cython: language_level=3

from cube import *
import random
from enum import Enum

gene_directions = ['R', 'L', 'U', 'D', 'C', 'B']
max_rotation = 3
solved_fitness = 65536 * 65536
class Mode(Enum):
    MAX = 1
    CUMULATED = 2
    LAST = 3

class Solution(object):
    def __init__(self, number_of_genes = 16):
        self.max_genes = number_of_genes
        self.genes = ""

    def calculate_fitness(self, actual_cube, test_chromosome, stop_at = -1, mode = Mode.MAX, color="B" ):
        _fitness = 0
        _cumulated_fitness = 0
        _last_fitness = 0
        _max_index = -1
        _print = False
        if stop_at == -1:
            stop_at = self.max_genes
        else:
            stop_at += 1
            _print = True
        for i in range(0,stop_at):
            _act_gene = test_chromosome[i*2:(i+1)*2]
            #print(_act_gene)
            assert(_act_gene[0] in gene_directions)
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
            
            #actual_cube.test_cube()

            _solved_faces, _solved_color = actual_cube.count_solved_faces(color)

            if _solved_faces >= 6:
                print("\n\nSolved!!! {}. : {}".format(str(i+1), test_chromosome[:((i+1)*2)]), end="\n\n")
                #print solution
                actual_cube.print_colored()
                _fitness = solved_fitness - i
                _max_index = i
                return _max_index, _fitness

            #_solved_edges = actual_cube.count_solved_edges(color)
            _same_as_center = actual_cube.count_same_as_center()
            _triangles = actual_cube.count_triangles()

            _pieces_in_place = actual_cube.count_pieces_in_place()

            #_new_fitness = _solved_color * 10000 + _solved_faces * 24 + _solved_edges - i / 50
            _new_fitness = _pieces_in_place + _solved_faces + _triangles + _same_as_center - i / 100
            #_new_fitness = _pieces_in_place - i / 100

            if _fitness < _new_fitness:
                _fitness = _new_fitness
                _max_index = i
            _last_fitness = _new_fitness
            _cumulated_fitness += _new_fitness

        if _print:
            print("Stopped at:", i)
            actual_cube.print_colored()
        _returned_fitness = 0
        if mode == Mode.MAX:
            _returned_fitness = _fitness
        elif mode == Mode.LAST:
            _returned_fitness = _last_fitness
        elif mode == Mode.CUMULATED:
            _returned_fitness = _cumulated_fitness
        else:
            assert( mode in Mode )
        return _max_index, _returned_fitness

    def generate_random_genes(self):
        _genes = []
        for i in range(0,self.max_genes):
            _genes.append(
                "{}{}".format(
                    random.sample( gene_directions, 1 )[0],
                    random.randint( 1, max_rotation )
                    )
                )
        self.set_chromosome( ''.join(_genes) )
        #print("Random generated genes:", self.genes)

    def mutate_genes(self, rate = 1):
        _index = random.randint( 0, self.max_genes - 1 )
        _new_genes = ""
        _mutation = "{}{}".format(
            random.sample( gene_directions, 1 )[0],
            random.randint( 1, max_rotation )
        )
        if _index > 0:
            _new_genes += self.genes[:(_index * 2)]
        _new_genes += _mutation
        _len = len(_new_genes) / 2
        if _len < self.max_genes:
            _new_genes += self.genes[( (_index + 1) * 2):]

        _len = len(_new_genes) / 2
        if _len != self.max_genes:
            print("{}. > {} > {}".format(_index, _new_genes, len(_new_genes) / 2))
        assert( (len(_new_genes) / 2) == self.max_genes )

        if rate > 1:
            for i in range(1,rate):
                _index = random.randint( 0, self.max_genes - 1 )
                _newer_genes = ""
                _mutation = "{}{}".format(
                    random.sample( gene_directions, 1 )[0],
                    random.randint( 1, max_rotation )
                )
                if _index > 0:
                    _newer_genes += _new_genes[:(_index * 2)]
                _newer_genes += _mutation
                _len = len(_newer_genes) / 2
                if _len < self.max_genes:
                    _newer_genes += _new_genes[( (_index + 1) * 2):]

                _len = len(_newer_genes) / 2
                if _len != self.max_genes:
                    print("{}. > {} > {}".format(_index, _newer_genes, len(_newer_genes) / 2))
                assert( (len(_newer_genes) / 2) == self.max_genes )
                _new_genes = _newer_genes

        return _new_genes

    def get_chromosome(self):
        return self.genes
    
    def simplify_chromosome(self, chromosome):
        return chromosome
        length = int(len(chromosome)/2)
        return_chromosome = chromosome
        if length < 2:
            return chromosome
        i = 1
        while i < length:
            _prev_gene = return_chromosome[(i-1)*2:i*2]
            _act_gene = return_chromosome[i*2:(i+1)*2]
            _shortend = False
            _samelentgh = False
            if _prev_gene[0] == _act_gene[0]:
                _count = ( int(_prev_gene[1]) + int(_act_gene[1]) ) % ( max_rotation + 1)
                if _count == 0:
                    return_chromosome = return_chromosome[:(i-1)*2] + return_chromosome[(i+1)*2:]
                    _shortend = True
                else:
                    return_chromosome = return_chromosome[:(i-1)*2] + "{}{}".format( _prev_gene[0], _count ) + return_chromosome[(i+1)*2:]
                    _samelentgh = True
            length = int(len(return_chromosome)/2)
            if _shortend:
                i -= 1
            elif not _samelentgh:
                i += 1
        return return_chromosome

    def set_chromosome(self, new_chromosome, simplify = False):
        self.genes = new_chromosome
        while (len(self.genes) / 2) < self.max_genes:
            self.genes += "{}{}".format(
                    random.sample( gene_directions, 1 )[0],
                    random.randint( 1, max_rotation )
                )
        if not simplify:
            return
        for i in range(1,self.max_genes):
            _prev_gene = self.genes[(i-1)*2:i*2]
            _act_gene = self.genes[i*2:(i+1)*2]
            if _prev_gene[0] == _act_gene[0]:
                _count = ( int(_prev_gene[1]) + int(_act_gene[1]) ) % ( max_rotation + 1)
                if _count == 0:
                    self.genes = self.genes[:(i-1)*2] + self.genes[(i+1)*2:]
                else:
                    self.genes = self.genes[:(i-1)*2] + "{}{}".format( _prev_gene[0], _count ) + self.genes[(i+1)*2:]
            while (len(self.genes) / 2) < self.max_genes:
                self.genes += "{}{}".format(
                        random.sample( gene_directions, 1 )[0],
                        random.randint( 1, max_rotation )
                    )
        assert( (len(self.genes) / 2) == self.max_genes )