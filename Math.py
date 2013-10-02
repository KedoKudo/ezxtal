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

    def __getattr__(self, item):
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

    def vector_length(self):
        """Return the length of a vector"""
        vector_length = 0
        for i in range(len(self)):
            vector_length += self._coords[i] ** 2
        vector_length = np.sqrt(vector_length)
        return vector_length


def find_angle(sin_val, cos_val):
        """
        find angle in radians based on value of sine and cosine
        :param sin_val: sin(theta)
        :param cos_val: cos(theta)
        """
        sin_val = float(sin_val)
        cos_val = float(cos_val)
        if sin_val >= 0.0:
            theta = np.arccos(cos_val)
        elif (sin_val < 0.0) & (cos_val < 0.0):
            theta = np.pi + np.arctan(sin_val/cos_val)
        elif (sin_val < 0.0) & (cos_val == 0.0):
            theta = 1.5*np.pi
        else:
            theta = 2*np.pi + np.arctan(sin_val/cos_val)
        return theta


def debug():
    """
    For debug purpose
    """
    print "Debug begins"

if __name__ == "__main__":
    debug()