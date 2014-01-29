#!/usr/bin/env python

__author__ = "KenZ"


#__Developer Note:
#   Provide some basic 2D plot types using PIL

import numpy as np
import Image
import ImageDraw
import matplotlib.pyplot as plt


class PoleFigure(object):
    """ generate standard pole figure based on given data, use matplotlib for backend"""

    def __init__(self, plot_data):
        """ initialize plot """
        pass


class GridPlot(object):
    """ Generate OIM/TSL like plots with grids (cube/hex) """

    def __init__(self, plot_data, grid="hex"):
        """ grid type:  ["hex", "cube"] """
        pass


def debug():
    """ module testing """
    print "Debug starts"


if __name__ == "__main__":
    debug()