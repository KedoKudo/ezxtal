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
# TODO: Though the calculation for reciprocal lattice is based on wiki, it somehow causes more noises than necessary.
# TODO: Need more testing on this subject


##
# Define lattice
class Lattice(object):
    """light weight class to represent the lattice structure"""
    __slots__ = ["__a", "__b", "__c", "__alpha", "__beta", "__gamma"]  # streamline memory use

    def __init__(self, a=Vector(3), b=Vector(3), c=Vector(3), alpha=None, beta=None, gamma=None):
        """initializing lattice instance with primitive lattice. can be a simple skeleton if the lattice constants are
        in reciprocal space

        @param: a: lattice constant a,
                b: lattice constant b,
                c: lattice constant c,
                alpha: angle between b and c;
                beta: angle between c and a;
                gamma: angle between a and b
        """
        self.__a = a
        self.__b = b
        self.__c = c
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma

    def __str__(self):
        """formatted output for lattice"""
        out_string = "lattice constants: a = {},\tb = {},\tc = {},\t".format(self.__a, self.__b, self.__c)
        out_string += "alpha = {},\tbeta = {}\tgamma = {}".format(self.__alpha, self.__beta, self.__gamma)
        return out_string

    @property
    def primitive(self):
        """return a dict of lattice constant in real space (primitive lattice)"""
        return {"a": self.__a, "b": self.__b, "c": self.__c, "alpha": self.__alpha, "beta": self.__beta,
                "gamma": self.__gamma}

    @primitive.setter
    def primitive(self, value):
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
    def reciprocal(self):
        """return a dict of lattice constants in reciprocal space"""
        ##
        # use physics definition of the reciprocal lattice
        # ref: http://en.wikipedia.org/wiki/Reciprocal_lattice
        lattice_real = np.zeros((3, 3))
        lattice_real[0] = self.__a
        lattice_real[1] = self.__b
        lattice_real[2] = self.__c
        lattice_real = lattice_real.T  # need to be a column vector based matrix
        lattice_recip = 2 * np.pi * La.inv(lattice_real)
        astar = lattice_recip[0]
        bstar = lattice_recip[1]
        cstar = lattice_recip[2]
        alpha_star = np.arccos(np.dot(bstar, cstar) / (La.norm(bstar) * La.norm(cstar))) * 180 / np.pi
        beta_star = np.arccos(np.dot(astar, cstar) / (La.norm(astar) * La.norm(cstar))) * 180 / np.pi
        gamma_star = np.arccos(np.dot(astar, bstar) / (La.norm(astar) * La.norm(bstar))) * 180 / np.pi
        return {"a*": astar, "b*": bstar, "c*": cstar, "alpha": alpha_star, "beta": beta_star, "gamma": gamma_star}

    @reciprocal.setter
    def reciprocal(self, value):
        """set lattice constants using reciprocal lattice constants

        value = {"a*": astar, "b*": bstar, "c*": cstar, "alpha": alpha_star, "beta": beta_star, "gamma": gamma_star}
        """
        lattice_recip = np.zeros((3, 3))
        lattice_recip[0] = value["a*"]
        lattice_recip[1] = value["b*"]
        lattice_recip[2] = value["c*"]  # stacking by row is the same as transpose
        lattice_real = 2 * np.pi * La.inv(lattice_recip)
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
    a = [1, 0, 0]
    b = [1, 1, 0]
    c = [0, 0, 0.5]
    alpha = 90
    beta = 90
    gamma = 90
    lattice_1 = Lattice(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    print lattice_1
    print lattice_1.reciprocal
    temp = lattice_1.reciprocal
    lattice_2 = Lattice()
    lattice_2.reciprocal = temp
    print lattice_2


if __name__ == "__main__":
    debug()
