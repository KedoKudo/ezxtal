#!/usr/bin/env python

__author__ = "KenZ"

#__Developer Note:
#   This is a simple module that contains some very useful math functions regarding calculation


import numpy as np


class Vector:
    """Multidimensinal vector class"""
    def __init__(self, d):
        """Initialize with d-dimensional vector of zeros"""
        self._coords = [0] * d

    def __len__(self):
        """Return the dimension of the vector"""
        return len(self._coords)

    def __getitem__(self, item):
        """Return item_th coordinate of the vector"""
        return self._coords[item]

    def __setitem__(self, key, value):
        """Set key_th coordinate with the given value"""
        self._coords[key] = value

    def __add__(self, other):
        """Return the sum of two vectors"""
        if len(self) != len(other):
            raise ValueError("dimension must agree")
        result = Vector(len(self))
        for i in range(len(self)):
            result[i] = self[i] + other[i]
        return result

    def __eq__(self, other):
        """Return true if two vector are the same"""
        if len(self) != len(other):
            raise ValueError("dimension must agree")
        return self._coords == other.__coords

    def __ne__(self, other):
        """Return True if vector differs from the other"""
        return not self == other

    def __str__(self):
        """Produce a string representation of vector"""
        return "<" + str(self._coords)[1:-1] + ">"  # remove the '[' ']' in the str(list)

    def vector_norm(self):
        """Return the length of a vector"""
        vector_norm = 0
        for i in range(len(self)):
            vector_norm += self._coords[i] ** 2
        vector_length = np.sqrt(vector_norm)
        return vector_length

    @property
    def coord(self):
        """new class style for a getter"""
        return self._coords

    @coord.setter
    def coord(self, value):
        """new style setter"""
        if len(self) != len(value):
            raise ValueError("dimension must agree")
        self._coords = value

    @coord.deleter
    def coord(self):
        """new style deleter"""
        del self._coords


def find_angle(sin_val, cos_val):
        """find angle in radians based on value of sine and cosine

        :param sin_val: sin(theta)
        :param cos_val: cos(theta)
        """
        sin_val = float(sin_val)
        cos_val = float(cos_val)
        if np.absolute(sin_val) <= 1e-6:
            theta = 0.0 if cos_val > 0.0 else np.pi  # special case: x-axis
        elif sin_val > 1e-6:
            theta = np.arccos(cos_val)  # 1 & 2 Quadrant
        else:
            theta = 2*np.pi - np.arccos(cos_val)  # 3 & 4 Quadrant
        return theta


def delta(i, j):
    """ return 1 if i,j are different, 0 otherwise """
    return 0.0 if np.absolute(i - j) < 1e-6 else 1.0


def levi_civita(i, j, k):
    """ return the output of Levi-Civita """
    return int((i - j) * (j - k) * (k - i) / 2)  # source: https://gist.github.com/Juanlu001/2689795


def debug():
    """
    For debug purpose
    """
    print "Debug begins"

if __name__ == "__main__":
    debug()