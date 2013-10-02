#!/usr/bin/env python

__author__ = "KenZ"

#__Developer Note:
#   This is a simple module that contains some very useful math functions regarding calculation


import numpy as np


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