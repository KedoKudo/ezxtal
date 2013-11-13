#!/usr/bin/env python

__author__ = "KenZ"

#__Developer Note:
#  This module is created for defining a lattice in a crystal and performing strain refinement based on the lattice
#  constants (ref: Beamline 34-ID-E, APS, ANL).
#  Only the deviatoric component of the strain is valid in the output of the strain refinement.

##
# import external library
from ezxtal.Math import Vector
import numpy as np
import numpy.linalg as La


##
# Define lattice
class Lattice(object):
    """light weight class to represent the lattice structure"""
    __slots__ = ["__a", "__b", "__c", "__alpha", "__beta", "__gamma"]  # streamline memory use

    def __init__(self):
        """ empty constructor """
        self.__a = Vector(3)
        self.__b = Vector(3)
        self.__c = Vector(3)
        self.__alpha = None
        self.__beta = None
        self.__gamma = None

    def __str__(self):
        """formatted output for lattice"""
        out_string = "lattice constants: a = {},\tb = {},\tc = {},\talpha = {},\tbeta = {}\tgamma = {}".format()
        return out_string

    @property
    def lattice(self):
        """return a dict of lattice constant in real space"""
        return {"a": self.__a, "b": self.__b, "c": self.__c, "alpha": self.__alpha, "beta": self.__beta,
                "gamma": self.__gamma}

    @lattice.setter
    def lattice(self, value):
        """set lattice constant

        value = {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}
        """
        self.__a = value["a"]
        self.__b = value["b"]
        self.__c = value["c"]
        self.__alpha = value["alpha"]
        self.__beta = value["beta"]
        self.__gamma = value["gamma"]

    @property
    def lattice_recip(self):
        """return a dict of lattice constants in reciprocal space"""
        ##
        # use physics definition of the reciprocal lattice
        # ref: http://en.wikipedia.org/wiki/Reciprocal_lattice
        lattice_real = np.zeros((3, 3))
        lattice_real[0] = self.__a
        lattice_real[1] = self.__b
        lattice_real[2] = self.__c
        lattice_real = lattice_real.T  # need to be a column vector based matrix
        lattice_recip = 2*np.pi * La.inv(lattice_real)
        astar = lattice_recip[0]
        bstar = lattice_recip[1]
        cstar = lattice_recip[2]
        alpha_star = np.arccos(np.dot(bstar, cstar) / (La.norm(bstar) * La.norm(cstar))) * 180 / np.pi
        beta_star = np.arccos(np.dot(astar, cstar) / (La.norm(astar) * La.norm(cstar))) * 180 / np.pi
        gamma_star = np.arccos(np.dot(astar, bstar) / (La.norm(astar) * La.norm(bstar))) * 180 / np.pi
        return {"a*": astar, "b*": bstar, "c*": cstar, "alpha": alpha_star, "beta": beta_star, "gamma": gamma_star}

    @lattice_recip.setter
    def lattice_recip(self, value):
        """set lattice constants using reciprocal lattice constants

        value = {"a*": astar, "b*": bstar, "c*": cstar, "alpha": alpha_star, "beta": beta_star, "gamma": gamma_star}
        """
        lattice_recip = np.zeros((3, 3))
        lattice_recip[0] = value["a*"]
        lattice_recip[1] = value["b*"]
        lattice_recip[2] = value["c*"]  # stacking by row is the same as transpose
        lattice_real = 1 / np.pi * La.inv(lattice_recip)
        self.__a = lattice_real.T[0]
        self.__b = lattice_real.T[1]
        self.__c = lattice_real.T[2]
        self.__alpha = np.arccos(np.dot(self.__b, self.__c) / (La.norm(self.__b) * La.norm(self.__c))) * 180 / np.pi
        self.__beta = np.arccos(np.dot(self.__a, self.__c) / (La.norm(self.__a) * La.norm(self.__c))) * 180 / np.pi
        self.__gamma = np.arccos(np.dot(self.__a, self.__b) / (La.norm(self.__a) * La.norm(self.__b))) * 180 / np.pi


##
# set up unit test
def debug():
    print "Module test begins"


if __name__ == "__main__":
    debug()
