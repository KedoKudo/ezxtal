#!/usr/local/env python

"""Contain slip system information."""

import sys


class SlipSystem(object):
    """ a container class that holds slip systems
    for various material.
    """

    def get_data(self, xtal='hcp', c_over_a=1.58):
        """
        return a list of slip systems based on given slip system with
        data structure [[slip plane], [slip direction]]
        """
        #NOTE: The data structure here is different from DAMASK convention.
        #      It is designed to follow Dr. Bieler's convention for describing
        #      slip systems.
        data = []  # list is more economic than dict
        if xtal == 'hcp':
            # Basal Slip {0001}<1120>
            data[0] = [[0, 0, 0, 1], [2, -1, -1, 0]]
            data[1] = [[0, 0, 0, 1], [-1, 2, -1, 0]]
            data[2] = [[0, 0, 0, 1], [-1, -1, 2, 0]]
            # Prism Slip {1010}<1120>
            data[3] = [[0, 1, -1, 0], [2, -1, -1, 0]]
            data[4] = [[-1, 0, 1, 0], [-1, 2, -1, 0]]
            data[5] = [[1, -1, 0, 0], [-1, -1, 2, 0]]
            # 2nd Prism Slip {1120}<1010>
            data[6] = [[2, -1, -1, 0], [0, 1, -1, 0]]
            data[7] = [[-1, 2, -1, 0], [-1, 0, 1, 0]]
            data[8] = [[-1, -1, 2, 0], [1, -1, 0, 0]]
            # Pyramidal a Slip {1011}<1120>
            data[9] = [[0, 1, -1, 1], [2, -1, -1, 0]]
            data[10] = [[-1, 1, 0, 1], [1, 1, -2, 0]]
            data[11] = [[-1, 0, 1, 1], [-1, 2, -1, 0]]
            data[12] = [[0, -1, 1, 1], [-2, 1, 1, 0]]
            data[13] = [[1, -1, 0, 1], [-1, -1, 2, 0]]
            data[14] = [[1, 0, -1, 1], [1, -2, 1, 0]]
            # Pyramidal c+a Slip {1011}<2113>
            data[15] = [[0, 1, -1, 1], [-1, 2, -1, -3]]
            data[16] = [[0, 1, -1, 1], [1, 1, -2, -3]]
            data[17] = [[-1, 1, 0, 1], [-2, 1, 1, -3]]
            data[18] = [[-1, 1, 0, 1], [-1, 2, -1, -3]]
            data[19] = [[-1, 0, 1, 1], [-1, -1, 2, -3]]
            data[20] = [[-1, 0, 1, 1], [-2, 1, 1, -3]]
            data[21] = [[0, -1, 1, 1], [1, -2, 1, -3]]
            data[22] = [[0, -1, 1, 1], [-1, -1, 2, -3]]
            data[23] = [[1, -1, 0, 1], [2, -1, -1, -3]]
            data[24] = [[1, -1, 0, 1], [1, -2, 1, -3]]
            data[25] = [[1, 0, -1, 1], [1, 1, -2, -3]]
            data[26] = [[1, 0, -1, 1], [2, -1, -1, -3]]
        else:
            print "will support in the future..."
            sys.exit(-1)
        return data