#!/usr/bin/env python

__author__ = "KenZ"

#__Developer Note:
#  This module is created for defining a lattice in a crystal and performing strain refinement based on the lattice
#  constants (ref: Beamline 34-ID-E, APS, ANL).
#  Only the deviatoric component of the strain is valid in the output of the strain refinement.

##
# import external library
from ezxtal.Math import Vector


##
# Define lattice
class Lattice(object):
    """
    light weight class to represent the lattice structure
    """
    def __init__(self):
        """ empty constructor """
        self._a = Vector(3)
        self._b = Vector(3)
        self._c = Vector(3)


##
# set up unit test
def debug():
    pass


if __name__ == "__main__":
    debug()
