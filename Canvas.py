#!/usr/bin/env python

__author__ = "KenZ"


#__Developer Note:
#   Provide some basic 2D plot types using PIL/svgwrite
#

#__NOTE:
#   Generally a vector plot is better, especially when integrated with
#   latex for auto-report.

import numpy as np
import svgwrite


class PoleFigure(object):
    """ generate standard pole figure based on given data
        NOTE: use svgwrite to generate a svg (vector plot) """

    def __init__(self, plot_data):
        """ initialize plot """
        pass


class GridPlot(object):
    """ Generate OIM/TSL like plots with grids (cube/hex) """

    def __init__(self, plot_data, grid="hex"):
        """ grid type:  ["hex", "cube"] """
        pass