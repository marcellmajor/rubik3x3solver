from solution import *
import colorama
#import sty
import copy
from collections import defaultdict

block_size = 3
blocks_per_face = block_size * block_size

colors = [
    "Y", # Yellow
    "R", # Red
    "W", # White
    "O", # Orange
    "G", # Green
    "B"  # Blue
]

color_selector = {
    "Y":"\33[48;5;11m", # Yellow
    "R":colorama.Back.RED, # Red
    "W":"\33[48;5;15m", # White
    "O":"\33[48;5;214m", # Orange
    "G":colorama.Back.GREEN, # Green
    "B":colorama.Back.BLUE  # Blue
}

class Cube(object):
    def __init__( self, top_face, row_of_faces, bottom_face ):
        self.top_face = top_face
        self.row_of_faces = row_of_faces
        self.bottom_face = bottom_face

    def rotate_right(self, count):
        for i in range(0,count):
            _factor = (2*block_size)
            _temp_face_third = self.row_of_faces[3][_factor:]
            self.row_of_faces[3] = self.row_of_faces[3][:_factor] + self.row_of_faces[2][_factor:]
            self.row_of_faces[2] = self.row_of_faces[2][:_factor] + self.row_of_faces[1][_factor:]
            self.row_of_faces[1] = self.row_of_faces[1][:_factor] + self.row_of_faces[0][_factor:]
            self.row_of_faces[0] = self.row_of_faces[0][:_factor] + _temp_face_third
            _temp_face = self.bottom_face
            self.bottom_face = _temp_face[6] + _temp_face[3] +_temp_face[0] + _temp_face[7] + _temp_face[4] +_temp_face[1] + _temp_face[8] + _temp_face[5] +_temp_face[2]

    def rotate_left(self, count):
        for i in range(0,count):
            _temp_face = self.top_face
            self.top_face = _temp_face[6] + _temp_face[3] +_temp_face[0] + _temp_face[7] + _temp_face[4] +_temp_face[1] + _temp_face[8] + _temp_face[5] +_temp_face[2]
            _factor = block_size
            _temp_face_third = self.row_of_faces[0][:_factor]
            self.row_of_faces[0] = self.row_of_faces[1][:_factor] + self.row_of_faces[0][_factor:]
            self.row_of_faces[1] = self.row_of_faces[2][:_factor] + self.row_of_faces[1][_factor:]
            self.row_of_faces[2] = self.row_of_faces[3][:_factor] + self.row_of_faces[2][_factor:]
            self.row_of_faces[3] = _temp_face_third + self.row_of_faces[3][_factor:]

    def rotate_up(self, count):
        for i in range(0,count):
            _temp_face = self.row_of_faces[2]
            self.row_of_faces[2] = self.row_of_faces[2][:2] + self.bottom_face[2] + self.row_of_faces[2][3:5] + self.bottom_face[5] + self.row_of_faces[2][6:8] + self.bottom_face[8]
            self.bottom_face = self.bottom_face[:2] + self.row_of_faces[0][6] + self.bottom_face[3:5] + self.row_of_faces[0][3] + self.bottom_face[6:8] + self.row_of_faces[0][0]
            self.row_of_faces[0] = self.top_face[8] + self.row_of_faces[0][1:3] + self.top_face[5] + self.row_of_faces[0][4:6] + self.top_face[2] + self.row_of_faces[0][7:9]
            self.top_face = self.top_face[:2] + _temp_face[2] + self.top_face[3:5] + _temp_face[5] + self.top_face[6:8] + _temp_face[8]
            _temp_face = self.row_of_faces[3]
            self.row_of_faces[3] = _temp_face[6] + _temp_face[3] +_temp_face[0] + _temp_face[7] + _temp_face[4] +_temp_face[1] + _temp_face[8] + _temp_face[5] +_temp_face[2]

    def rotate_down(self,count):
        for i in range(0,count):
            _temp_face = self.row_of_faces[1]
            self.row_of_faces[1] = _temp_face[6] + _temp_face[3] +_temp_face[0] + _temp_face[7] + _temp_face[4] +_temp_face[1] + _temp_face[8] + _temp_face[5] +_temp_face[2]
            _temp_face = self.row_of_faces[2]
            self.row_of_faces[2] = self.top_face[0] + self.row_of_faces[2][1:3] + self.top_face[3] + self.row_of_faces[2][4:6] + self.top_face[6] + self.row_of_faces[2][7:9]
            self.top_face = self.row_of_faces[0][8] + self.top_face[1:3] + self.row_of_faces[0][5] + self.top_face[4:6] + self.row_of_faces[0][2] + self.top_face[7:9]
            self.row_of_faces[0] = self.row_of_faces[0][:2] + self.bottom_face[6] + self.row_of_faces[0][3:5] + self.bottom_face[3] + self.row_of_faces[0][6:8] + self.bottom_face[0]
            self.bottom_face =  _temp_face[0] + self.bottom_face[1:3] + _temp_face[3] + self.bottom_face[4:6] + _temp_face[6] + self.bottom_face[7:9]

    def rotate_cloclkwise(self, count):
        for i in range(0,count):
            _temp_face = self.row_of_faces[2]
            self.row_of_faces[2] = _temp_face[6] + _temp_face[3] +_temp_face[0] + _temp_face[7] + _temp_face[4] +_temp_face[1] + _temp_face[8] + _temp_face[5] +_temp_face[2]
            _temp_face = self.top_face
            self.top_face = self.top_face[:6] + self.row_of_faces[1][8] + self.row_of_faces[1][5] + self.row_of_faces[1][2]
            self.row_of_faces[1] = self.row_of_faces[1][:2] + self.bottom_face[0] + self.row_of_faces[1][3:5] + self.bottom_face[1] + self.row_of_faces[1][6:8] + self.bottom_face[2]
            self.bottom_face = self.row_of_faces[3][6] + self.row_of_faces[3][3] + self.row_of_faces[3][0] + self.bottom_face[3:]
            self.row_of_faces[3] = _temp_face[6] + self.row_of_faces[3][1:3] + _temp_face[7] + self.row_of_faces[3][4:6] + _temp_face[8] + self.row_of_faces[3][7:9]

    def rotate_backwards(self, count):
        for i in range(0,count):
            _temp_face = self.row_of_faces[0]
            self.row_of_faces[0] = _temp_face[6] + _temp_face[3] +_temp_face[0] + _temp_face[7] + _temp_face[4] +_temp_face[1] + _temp_face[8] + _temp_face[5] +_temp_face[2]
            _temp_face = self.top_face
            self.top_face = self.row_of_faces[3][2] + self.row_of_faces[3][5] +self.row_of_faces[3][8] +self.top_face[3:]
            self.row_of_faces[3] = self.row_of_faces[3][:2] + self.bottom_face[8] + self.row_of_faces[3][3:5] + self.bottom_face[7] + self.row_of_faces[3][6:8] + self.bottom_face[6]
            self.bottom_face = self.bottom_face[:6] + self.row_of_faces[1][0] + self.row_of_faces[1][3] + self.row_of_faces[1][6]
            self.row_of_faces[1] = _temp_face[2] + self.row_of_faces[1][1:3] + _temp_face[1] + self.row_of_faces[1][4:6] + _temp_face[0] + self.row_of_faces[1][7:9]

    def count_solved_faces( self, color="" ):
        faces = [
            self.top_face,
            self.row_of_faces[0],
            self.row_of_faces[1],
            self.row_of_faces[2],
            self.row_of_faces[3],
            self.bottom_face
            ]
        _counter = 0
        _color_solved = 0
        for _face in faces:
            if _face[0] == _face[1] == _face[2] == _face[3] == _face[4] == _face[5] == _face[6] == _face[7] == _face[8]:
                _counter += 1
                if _face[0] == color:
                    _color_solved = 1
        return _counter, _color_solved

    def count_solved_edges( self, color="" ):
        faces = [
            self.top_face,
            self.row_of_faces[0],
            self.row_of_faces[1],
            self.row_of_faces[2],
            self.row_of_faces[3],
            self.bottom_face
            ]
        _ret_counter = 0
        for _face in faces:
            _counter = 0
            # if _face[0] == _face[2] == _face[6] == _face[8]:
            #     _counter += 1
            # if _face[1] == _face[3] == _face[4] == _face[5] == _face[7]:
            #     _counter += 1
            if _face[0] == _face[1] == _face[2]:
                _counter += 1
                if color != "" and _face[0] == color:
                    _counter += 3
            if _face[6] == _face[7] == _face[8]:
                _counter += 1
                if color != "" and _face[6] == color:
                    _counter += 3
            if _face[0] == _face[3] == _face[6]:
                _counter += 1
                if color != "" and _face[0] == color:
                    _counter += 3
            if _face[2] == _face[5] == _face[8]:
                _counter += 1
                if color != "" and _face[2] == color:
                    _counter += 3
            if _face[1] == _face[4] == _face[7]:
                _counter += 3
                if color != "" and _face[1] == color:
                    _counter += 3
            if _face[3] == _face[4] == _face[5]:
                _counter += 3
                if color != "" and _face[3] == color:
                    _counter += 3
            _ret_counter += _counter
        return _ret_counter

    def print_colored(self):
        colorama.init()
        print(' ' * block_size * 4,end='')
        print(color_selector[self.top_face[0]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL,end='')
        print(color_selector[self.top_face[1]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL,end='')
        print(color_selector[self.top_face[2]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL)
        print(' ' * block_size * 4,end='')
        print(color_selector[self.top_face[3]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL,end='')
        print(color_selector[self.top_face[4]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL,end='')
        print(color_selector[self.top_face[5]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL)
        print(' ' * block_size * 4,end='')
        print(color_selector[self.top_face[6]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL,end='')
        print(color_selector[self.top_face[7]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL,end='')
        print(color_selector[self.top_face[8]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL)

        for _face in self.row_of_faces:
            print(color_selector[_face[0]] + ' ' * 2,end='')
            print(color_selector[_face[1]] + ' ' * 2,end='')
            print(color_selector[_face[2]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL)
        for _face in self.row_of_faces:
            print(color_selector[_face[3]] + ' ' * 2,end='')
            print(color_selector[_face[4]] + ' ' * 2,end='')
            print(color_selector[_face[5]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL)
        for _face in self.row_of_faces:
            print(color_selector[_face[6]] + ' ' * 2,end='')
            print(color_selector[_face[7]] + ' ' * 2,end='')
            print(color_selector[_face[8]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL)

        print(' ' * block_size * 4,end='')
        print(color_selector[self.bottom_face[0]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL,end='')
        print(color_selector[self.bottom_face[1]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL,end='')
        print(color_selector[self.bottom_face[2]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL)
        print(' ' * block_size * 4,end='')
        print(color_selector[self.bottom_face[3]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL,end='')
        print(color_selector[self.bottom_face[4]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL,end='')
        print(color_selector[self.bottom_face[5]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL)
        print(' ' * block_size * 4,end='')
        print(color_selector[self.bottom_face[6]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL,end='')
        print(color_selector[self.bottom_face[7]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL,end='')
        print(color_selector[self.bottom_face[8]] + ' ' * 2,end='')
        print(colorama.Style.RESET_ALL)

    def test_cube(self):
        _colorcount = defaultdict(int)
        assert(len(self.top_face) == blocks_per_face)
        for _color in self.top_face:
            _colorcount[_color] += 1
            assert(_color in colors)
        assert(len(self.row_of_faces) == 4)
        for _face in self.row_of_faces:
            assert(len(_face) == blocks_per_face)
            for _color in _face:
                _colorcount[_color] += 1
                assert(_color in colors)
        assert(len(self.bottom_face) == blocks_per_face)
        for _color in self.bottom_face:
            _colorcount[_color] += 1
            assert(_color in colors)
        #print("Cube seems OK")
        for _color in _colorcount.keys():
            if _colorcount[_color] != blocks_per_face:
                print("{}:{}".format(_color,_colorcount[_color]))
                self.print_colored()
            assert( _colorcount[_color] == blocks_per_face )
            assert( _color in colors )
    
    def get_almost_complete_color(self, color):
        faces = [
            self.top_face,
            self.row_of_faces[0],
            self.row_of_faces[1],
            self.row_of_faces[2],
            self.row_of_faces[3],
            self.bottom_face
            ]
        _selected_color = color
        _color_count = defaultdict(int)
        for _face in faces:
            if _face[0] == _face[1] == _face[2] == _face[3] == _face[4] == _face[5] == _face[6] == _face[7] == _face[8]:
                continue
            if _face[0] == _face[1] == _face[2]:
                _color_count[_face[0]] += 1
            if _face[6] == _face[7] == _face[8]:
                _color_count[_face[6]] += 1
            if _face[0] == _face[3] == _face[6]:
                _color_count[_face[0]] += 1
            if _face[2] == _face[5] == _face[8]:
                _color_count[_face[2]] += 1
            if _face[1] == _face[4] == _face[7]:
                _color_count[_face[1]] += 2
            if _face[3] == _face[4] == _face[5]:
                _color_count[_face[3]] += 2
        for _color_act_index in range(0,len(colors)):
            if colors[_color_act_index] == max(_color_count, key=_color_count.get):
                _selected_color = _color_act_index
        return _selected_color

    def clone(self):
        return copy.deepcopy(self)