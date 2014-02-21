#!/usr/bin/env python

__author__ = "KenZ"


#__Developer Note:
#   Provide some basic 2D plot types using matplotlib
#

#__NOTE:
#   Use matplotlib, PIL does not support antialias

import numpy as np
import itertools
import matplotlib.pyplot as plt
from ezxtal import EulerAngle
from ezxtal import RotationMatrix


class PoleFigure(object):
    """ generate standard pole figure based on given data """

    def __init__(self, eulers_list):
        """ initialize plot, take in a list of Euler Angles """
        self.__dpi = 150
        self.__title = "default"
        self.__data = eulers_list
        self.__plane_list = [[0, 0, 1]]
        self.__is_literal = True  # whether to use permutation to get a family of planes
        self.__lattice_vector = np.array([1.0, 1.0, 1.0])  # most simple case as default
        self.__output = "pdf"
        self.__clr_list = None
        self.__ref = np.eye(3)  # matrix used to define xtal unit cell in reference configuration
        # set up pyplot
        self.__fig = plt.figure()
        self.__fig.add_subplot(111, aspect='equal')
        self.__fig.gca().add_artist(plt.Circle((0, 0), 1, color='k', fill=False))
        self.__unique_marker = False
        plt.plot([-1, 1], [0, 0], c="k")
        plt.plot([0, 0], [-1, 1], c="k")
        plt.gca().set_xlim((-1.15, 1.15))
        plt.gca().set_ylim((-1.15, 1.15))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

    @property
    def output(self):
        """ output format """
        return self.__output

    @output.setter
    def output(self, new_format):
        self.__output = new_format

    @property
    def is_literal(self):
        """ whether to plot by families (default=True) """
        return self.__is_literal

    @is_literal.setter
    def is_literal(self, new_state):
        self.__is_literal = new_state

    @property
    def unique_marker(self):
        return self.__unique_marker

    @unique_marker.setter
    def unique_marker(self, new_state):
        self.__unique_marker = new_state

    @property
    def lattice_vector(self):
        return self.__lattice_vector

    @lattice_vector.setter
    def lattice_vector(self, new_lattice):
        self.__lattice_vector = new_lattice

    @property
    def plane_list(self):
        """ return the family of the plane to be plotted """
        return self.__plane_list

    @plane_list.setter
    def plane_list(self, new_list):
        """ plot user specified set of planes instead of default one """
        self.__plane_list = new_list

    def add_plane(self, plane):
        """ add one family to the pole figure """
        self.__plane_list.append(plane)

    def set_ref(self, new_ref):
        """ set xtal unit cell in reference configuration """
        self.__ref = new_ref

    def set_color_list(self, new_list):
        """ use given list of colors (will recycle old color when reaching the end """
        self.__clr_list = itertools.cycle(new_list)

    @property
    def title(self):
        return str(self.__title)

    @title.setter
    def title(self, new_title):
        self.__title = new_title

    def plot(self):
        """ generate the plot and save it """
        #prepare the marker list
        marker = itertools.cycle((',', '+', '.', 'o', '*',
                                  '^', 'v', '<', '>', '8',
                                  's', 'p', 'h', 'H', 'D',
                                  'd'))
        # first categorised with plane
        for each_plane in self.plane_list:
            if self.is_literal:
                label = "[" + "{0} {1} {2}".format(each_plane[0], each_plane[1], each_plane[2]) + "]"
            else:
                label = "{"+"{0}, {1}, {2}".format(each_plane[0], each_plane[1], each_plane[2]) + "}"
            x_list = []
            y_list = []
            if self.is_literal:
                tmp = [each_plane]
                opposite_plane = [-item for item in each_plane]
                tmp.append(opposite_plane)
            else:
                tmp = PoleFigure.get_permutations(each_plane)
            # second categorised with grain ID
            my_marker = "."  # default marker
            for i in range(len(self.__data)):
                each_euler = self.__data[i]
                if self.unique_marker:
                    my_marker = marker.next()
                    plt.rcParams['text.usetex'] = False  # otherwise, '^' will cause trouble
                euler = EulerAngle(each_euler[0], each_euler[1], each_euler[2])
                rot_m = np.dot(self.__ref, euler.rotation_matrix)
                self.__data[i] = RotationMatrix(rot_m).euler_angle
                for each_pole in tmp:
                    tmp_pole = np.array(each_pole) / self.lattice_vector
                    tmp_pole /= np.linalg.norm(tmp_pole)
                    coord = np.dot(rot_m, tmp_pole)
                    if coord[2] < 0:
                        continue  # not pointing up, moving on
                    else:
                        x = coord[0] / (1.0 + float(coord[2]))
                        y = coord[1] / (1.0 + float(coord[2]))
                        # need to rotate 90 degree
                        x_list.append(y)
                        y_list.append(-x)
            # start plotting
            if self.__clr_list is not None:
                clr = self.__clr_list.next()
            else:
                clr = np.random.rand(3, 1)
            plt.scatter(x_list, y_list, marker=my_marker, c=clr, label=label, edgecolor='none')
        # label x/y axis
        plt.text(1.1, 0.0, "y", horizontalalignment='center', verticalalignment='center', fontsize=15)
        plt.text(0.0, -1.1, "x", horizontalalignment='center', verticalalignment='center', fontsize=15)
        # set legend
        plt.legend(loc='upper left', numpoints=1, ncol=6, fontsize=8, bbox_to_anchor=(0, 0))
        plt.title(self.title)
        plt.savefig(self.title + "." + self.output)
        plt.close()

    @classmethod
    def get_permutations(cls, plane):
        """ return a list of unique family of planes w.r.t the given plane """
        tmp = []
        for item in itertools.permutations(plane):
            tmp.append(tuple(item))
            for scale in itertools.permutations([-1, 1, 1]):
                tmp.append(tuple(np.array(item) * np.array(scale)))
            for scale in itertools.permutations([-1, -1, 1]):
                tmp.append(tuple(np.array(item) * np.array(scale)))
            tmp.append(tuple(-tmp_item for tmp_item in item))
        tmp = list(set(tmp))
        return tmp


class GridPlot(object):
    """ Generate OIM/TSL like plots with grids (cube/hex) """

    def __init__(self, plot_data, grid="hex"):
        """ grid type:  ["hex", "cube"] """
        pass