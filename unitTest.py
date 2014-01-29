#!/usr/bin/env python

##
# Unit test for ezxtal module using PET
#   url: http://docs.python.org/2/library/unittest.html


import unittest
from ezxtal.Orientation import EulerAngle
from ezxtal.Orientation import RotationMatrix
from ezxtal.Orientation import Quaternion


class TestEulerAngle(unittest.TestCase):
    """ Test EulerAngle class """
    print "Testing Euler Angle"


class TestRoationMaitrx(unittest.TestCase):
    """ Test RotationMatrix class """
    print "Testing Rotation Matrix"


class TestQuaternion(unittest.TestCase):
    """ Test Quaternion class """
    print "Testing Quaternion"


if __name__ == '__main__':
    unittest.main()
