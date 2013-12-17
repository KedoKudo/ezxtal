#!/usr/bin/env python

__author__ = "KenZ"

#__Developer Note:
#   A set of class used for describing orientation in space


import numpy as np
import numpy.linalg as La
import cmath
from Math import find_angle
from Math import delta
from Math import levi_civita
from abc import ABCMeta, abstractproperty


##
# MACRO
ERROR = 1e-12


##
#TODO: Need class for Quaternion
#TODO: There seems to be some bugs in the Rodrigues class, do not use it until it's fixed.
#


class XtalOrientation(object):
    """ Base class for all crystal orientation related class"""
    __metaclass__ = ABCMeta

    @abstractproperty
    def euler_angle(self):
        """ return orientation in the form of Euler angle"""
        return "Implemented in subclass"

    @abstractproperty
    def rotation_matrix(self):
        """ return orientation in the form of rotation matrix """
        return "Implemented in subclass"

    @abstractproperty
    def orientation_matrix(self):
        """ return orientation in the form of orientation matrix """
        return "Implemented in subclass"

    @abstractproperty
    def rotation_angle(self):
        """ return the rotation angle """
        return "Implemented in subclass"

    @abstractproperty
    def rotation_axis(self):
        """ return the rotation axis """
        return "Implemented in subclass"

    @abstractproperty
    def rodrigues(self):
        """ return the orientation in the form of Rodrigues vector """
        return "Implemented in subclass"

    @property
    def quaternion(self):
        """ return teh orientation in the form of Quaternion vector """
        return "Implemented in subclass"


class EulerAngle(XtalOrientation):
    """ Bunge Euler Angle (intrinsic rotation: z->x'->z'') """
    __slots__ = ["__phi1", "__phi", "__phi2"]  # streamline memory use

    def __init__(self, phi1, phi, phi2):
        """ store the Euler angle in radians """
        deg2rad = np.pi / 180
        self.__phi1 = phi1 * deg2rad
        self.__phi = phi * deg2rad
        self.__phi2 = phi2 * deg2rad

    def __str__(self):
        """ output Euler angles using degrees """
        rad2deg = 180 / np.pi
        out_string = "Bunge Euler angle: [{:.2f}, {:.2f}, {:.2f}]".format(self.__phi1 * rad2deg,
                                                                          self.__phi * rad2deg,
                                                                          self.__phi2 * rad2deg)
        return out_string

    def __len__(self):
        return 3

    @property
    def euler_angle(self):
        """ Euler angles in degrees """
        return [self.__phi1, self.__phi, self.__phi2]

    def set_euler_angle(self, new_angles):
        """ set new Euler angles """
        deg2rad = np.pi / 180
        self.__phi1 = new_angles[0] * deg2rad
        self.__phi = new_angles[1] * deg2rad
        self.__phi2 = new_angles[2] * deg2rad

    @property
    def phi1(self):
        """ accessor for the first Euler angle """
        return self.__phi1

    @property
    def phi(self):
        """ accessor for second Euler angle """
        return self.__phi

    @property
    def phi2(self):
        """ accessor for last Euler angle """
        return self.__phi2

    @property
    def orientation_matrix(self):
        """ return orientation matrix based Bunge Euler angle"""
        ##
        # Note: Orientation matrix (g) is the transpose of teh standard rotation matrix generally defined in Math and
        #       many other fields. The reason for material science people to use this approach is because Bunge was more
        #       interested in converting sample (global) coordinate to crystal (local). Since a lot of material science
        #       people are accustomed to the Bunge system, which is based on orientation matrix instead of the more
        #       commonly accepted rotation matrix (R), g is generally computed instead of R during computation material
        #       science.
        g_phi1 = np.array([[np.cos(self.__phi1), np.sin(self.__phi1), 0.0],
                           [-np.sin(self.__phi1), np.cos(self.__phi1), 0.0],
                           [0.0, 0.0, 1.0]])
        g_phi = np.array([[1.0, 0.0, 0.0],
                          [0.0, np.cos(self.__phi), np.sin(self.__phi)],
                          [0.0, -np.sin(self.__phi), np.cos(self.__phi)]])
        g_phi2 = np.array([[np.cos(self.__phi2), np.sin(self.__phi2), 0.0],
                           [-np.sin(self.__phi2), np.cos(self.__phi2), 0.0],
                           [0.0, 0.0, 1.0]])
        g = np.dot(g_phi2, np.dot(g_phi, g_phi1))  # the total orientation matrix is g = g_phi2 * g_phi * g_phi1
        for i in range(3):
            for j in range(3):
                if abs(g[i, j]) < ERROR:
                    g[i, j] = 0.0
        return g

    @property
    def rotation_matrix(self):
        """ return the standard rotation associated with the Euler angle"""
        return self.orientation_matrix.T

    @property
    def rotation_axis(self):
        """ return the rotation axis """
        rot_m = RotationMatrix(self.rotation_matrix)
        return rot_m.rotation_axis

    @property
    def rotation_angle(self):
        """ return the rotation angle around the rotation axis """
        rot_m = RotationMatrix(self.rotation_matrix)
        return rot_m.rotation_angle

    @property
    def rodrigues(self):
        """ convert Euler angles into Rodrigues vector """
        return RotationMatrix(self.rotation_matrix).rodrigues

    #TODO: add conversion to quaternion


class RotationMatrix(XtalOrientation):
    """ represent the orientation using standard rotation matrix """
    __slots__ = ["__r"]

    def __init__(self, rotation_matrix):
        """ initialize rotation matrix with a 3x3 numpy array """
        self.__r = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                self.__r[i, j] = rotation_matrix[i, j]

    def __len__(self):
        """ size of the rotation matrix"""
        return "3 x 3"

    @property
    def rotation_matrix(self):
        """ accessor for rotation matrix """
        return self.__r

    def set_rotation_matrix(self, new_r):
        """ modifier for rotation matrix"""
        self.__r = new_r

    @property
    def orientation_matrix(self):
        """ return the orientation matrix """
        return self.__r.T

    @property
    def rotation_axis(self):
        values, axis = La.eig(self.__r)
        axis = axis.T
        # The rotation axis can only be found by analyze the e^{-i\theta} component.
        # For counter clockwise rotation by theta, the rotation axis is the cross
        # product of v_r and v_i from e^{-i\theta}
        rot_axis = "ERROR"
        for index in range(3):
            if values[index].imag < 0:
                v_r = [item.real for item in axis[index]]
                v_i = [item.imag for item in axis[index]]
                rot_axis = np.cross(v_r, v_i)
                rot_axis /= La.norm(rot_axis)
        # for non-rotation cases
        if values[0] == values[1] == values[2] == 1:
            rot_axis = np.array([1, 0, 0])  # arbitrary rotation axis
        return rot_axis

    @property
    def rotation_angle(self):
        """ Get rotation angle around the rotation axis w.r.t the result in self.rotation_axis """
        values, axis = La.eig(self.__r)
        # if it returns a rotation angle that is greater than 360, then something is wrong
        rot_angle = "ERROR"
        for index in range(3):
            if np.absolute(values[index] - 1) > 1e-6:  # 1 is eigenvalue for the rotation axis
                rot_angle = np.absolute(cmath.phase(values[index]) * 180.0 / np.pi)
                break
        if rot_angle == "ERROR":
            rot_angle = 0.0  # non-rotation case
        return rot_angle

    @property
    def euler_angle(self):
        """ return one set of Euler Angle w.r.t the rotation matrix """
        angles = [0, 0, 0]
        if np.absolute(self.__r[2, 2] - 1.0) < 1e-6:
        #simple one rotation case
            phi1 = find_angle(-self.__r[0, 1], self.__r[0, 0])
            angles[0] = phi1 * 180 / np.pi
        else:
            phi = np.arccos(self.__r[2, 2])
            angles[1] = phi * 180 / np.pi
            #calculate __phi1
            sin_phi1 = self.__r[0, 2] / np.sin(phi)
            cos_phi1 = -self.__r[1, 2] / np.sin(phi)
            phi1 = find_angle(sin_phi1, cos_phi1)
            angles[0] = phi1 * 180 / np.pi
            #calculate __phi2
            sin_phi2 = self.__r[2, 0] / np.sin(phi)
            cos_phi2 = self.__r[2, 1] / np.sin(phi)
            phi2 = find_angle(sin_phi2, cos_phi2)
            angles[2] = phi2 * 180 / np.pi
        return angles

    @property
    def rodrigues(self):
        """ convert rotation matrix into Rodrigues vector """
        scale = -1.0 / (1.0 + sum(sum(self.__r * self.__r)))
        rodrigues = [0, 0, 0]
        for i in range(3):
            rodrigues[i] = scale * sum([sum([levi_civita(i, j, k) * self.__r[j, k] for k in range(3)])
                                        for j in range(3)])
        return rodrigues

    #TODO: add conversion to quaternion


class Rodrigues(XtalOrientation):
    """ representing crystal orientation using Rodrigues vector"""
    __slots__ = ["__r"]

    def __init__(self, r1, r2, r3):
        """ initialize with  3 components"""
        self.__r = [r1, r2, r3]

    def __len__(self):
        """ the length of the vector """
        return 3

    @property
    def rodrigues(self):
        """ accessor for Rodrigues vector """
        return self.__r

    def set_rodrigues(self, new_r):
        """ modifier for Rodrigues vector """
        self.__r = new_r

    @property
    def rotation_matrix(self):
        """ convert Rodrigues vector to rotation matrix """
        rot_matrix = np.zeros((3, 3))
        scale = 1.0 / (1.0 + np.dot(self.__r, self.__r))
        for i in range(3):
            for j in range(3):
                rot_matrix[i, j] = scale * ((1 - np.dot(self.__r, self.__r)) * delta(i, j) +
                                            2 * self.__r[i] * self.__r[j] -
                                            2 * sum([levi_civita(i, j, k) * self.__r[k] for k in range(3)]))
        return rot_matrix

    @property
    def orientation_matrix(self):
        """ convert Rodrigues vector into orientation matrix """
        return self.rotation_matrix.T

    @property
    def euler_angle(self):
        """ convert Rodrigues vector into Euler angles """
        return RotationMatrix(self.rotation_matrix).euler_angle

    @property
    def rotation_axis(self):
        """ return the rotation axis """
        return self.__r / La.norm(self.__r)

    @property
    def rotation_angle(self):
        """ return the rotation angle around the rotation axis """
        return np.arctan(La.norm(self.__r))


def debug():
    """ Module debugging """
    print "Module debug begins:"
    eulerangle_1 = EulerAngle(40, 20, 5)
    print eulerangle_1.rotation_matrix
    rodrigues_1 = Rodrigues(eulerangle_1.rodrigues[0],
                            eulerangle_1.rodrigues[1],
                            eulerangle_1.rodrigues[2])
    print rodrigues_1.rotation_matrix  # something is very wrong here, I'll fix it later...


if __name__ == "__main__":
    debug()
