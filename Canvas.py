#!/usr/bin/env python

__author__ = "KenZ"


#__Developer Note:
#   Provide some basic 2D plot types using matplotlib
#

#__NOTE:
#   Use matplotlib, PIL does not support antialias

import matplotlib as mpl
mpl.use("cairo")

import numpy as np
import itertools
import matplotlib.pyplot as plt
from ezxtal import EulerAngle


class PoleFigure(object):
    """ generate standard pole figure based on given data """

    def __init__(self, eulers_list):
        """ initialize plot, take in a list of Euler Angles """
        self.__dpi = 150
        self.__title = "default"
        self.__data = eulers_list
        self.__plane_list = [[0, 0, 1]]
        self.__lattice_vector = np.array([1.0, 1.0, 1.0])  # most simple case as default
        # set up pyplot
        self.__fig = plt.figure()
        self.__fig.add_subplot(111, aspect='equal')
        self.__fig.gca().add_artist(plt.Circle((0, 0), 1, color='k', fill=False))
        self.__unique_marker = False
        plt.plot([-1, 1], [0, 0], c="k")
        plt.plot([0, 0], [-1, 1], c="k")
        plt.gca().set_xlim((-1.05, 1.05))
        plt.gca().set_ylim((-1.05, 1.05))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

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

    def add_plane(self, plane):
        """ add one family to the pole figure """
        self.__plane_list.append(plane)

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
            label = "{"+"{0}, {1}, {2}".format(each_plane[0], each_plane[1], each_plane[2]) + "}"
            tmp = []  # list of pole related to this family
            x_list = []
            y_list = []
            for item in itertools.permutations(each_plane):
                tmp.append(item)
            for item in itertools.permutations([-each_plane[0], each_plane[1], each_plane[2]]):
                tmp.append(item)
            for item in itertools.permutations([each_plane[0], -each_plane[1], each_plane[2]]):
                tmp.append(item)
            for item in itertools.permutations([each_plane[0], each_plane[1], -each_plane[2]]):
                tmp.append(item)
            for item in itertools.permutations([-each_plane[0], -each_plane[1], each_plane[2]]):
                tmp.append(item)
            for item in itertools.permutations([each_plane[0], -each_plane[1], -each_plane[2]]):
                tmp.append(item)
            for item in itertools.permutations([-tmp_item for tmp_item in each_plane]):
                tmp.append(item)
            tmp = list(set(tmp))  # remove duplicates
            # second categorised with grain ID
            my_marker = ","  # default marker
            for each_euler in self.__data:
                if self.unique_marker:
                    my_marker = marker.next()
                    plt.rcParams['text.usetex'] = False # otherwise, '^' will cause trouble
                euler = EulerAngle(each_euler[0], each_euler[1], each_euler[2])
                rot_m = euler.rotation_matrix
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
            plt.scatter(x_list, y_list, marker=my_marker, c=np.random.rand(3, 1), label=label)
        # set legend
        plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
        plt.title(self.title)
        plt.savefig(self.title + ".pdf")


class GridPlot(object):
    """ Generate OIM/TSL like plots with grids (cube/hex) """

    def __init__(self, plot_data, grid="hex"):
        """ grid type:  ["hex", "cube"] """
        pass