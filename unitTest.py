#!/usr/bin/env python

##
# Unit test using PET
#   url: http://docs.python.org/2/library/unittest.html
# need to work on this later


import unittest
from ezxtal.Orientation import EulerAngle
from ezxtal.Orientation import RotationMatrix
from ezxtal.Orientation import Quaternion


class TestEulerAngle(unittest.TestCase):
    """ Test EulerAngle class """

    def setUp(self):
        self.euler_list = [[0, 0, 0], [90, 0, 0], [0, 90, 0], [20, 80, 100]]


class TestRoationMaitrx(unittest.TestCase):
    """ Test RotationMatrix class """
    print "Testing Rotation Matrix"


class TestQuaternion(unittest.TestCase):
    """ Test Quaternion class """
    print "Testing Quaternion"


if __name__ == '__main__':
    unittest.main()
