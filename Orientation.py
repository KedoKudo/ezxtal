#!/usr/bin/env python

__author__ = "KenZ"

#__Developer Note:
#   A set of class used for describing orientation in space

import numpy as np
import numpy.linalg as La
import cmath
from Math import find_angle

##
#TODO: Need class for Quaternion, Rodrigues vector, and a wrapper class for XtalOrientation
#


class EulerAngle(object):
    """ Bunge Euler Angle (intrinsic rotation: z->x'->z'') """
    __slots__ = ["__phi1", "__phi", "__phi2"]  # streamline memory use

    def __init__(self, phi1, phi, phi2):
        """ store the Euler angle in radians """
        deg2rad = np.pi / 180
        self.__phi1 = phi1 * deg2rad
        self.__phi = phi * deg2rad
        self.__phi2 = phi2 * deg2rad

    def __eq__(self, other):
        """ crystal symmetry is not taken into consideration at this point """
        # do calculation here, Bunge Euler angle is not 1-on-1 mapping,
        # have to compare the Rotation matrix instead. No symmetry is
        # considered here as the crystal symmetry will have a huge effect
        # and should be considered in the Crystal module
        flag = False
        test = np.dot(self.rotation_axis, other.rotation_axis)
        if 1 - np.absolute(test) < 1e-6:
            # the rotation axis is the same, now considering rotation angle
            test_ang = self.rotation_angle - other.rotation_angle
            if np.absolute(test_ang) < 1e-6:
                flag = True
        return flag

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
    def angles(self):
        """ Euler angles in degrees """
        return [self.__phi1, self.__phi, self.__phi2]

    @angles.setter
    def angles(self, new_angles):
        """ set new Euler angles """
        self.__phi1 = new_angles[0]
        self.__phi = new_angles[1]
        self.__phi2 = new_angles[2]

    @property
    def phi1(self):
        """ the first Euler angle """
        return self.__phi1

    @phi1.setter
    def phi1(self, new_phi1):
        """ set new phi1 """
        self.__phi1 = new_phi1

    @property
    def phi(self):
        """ second Euler angle """
        return self.__phi

    @phi.setter
    def phi(self, new_phi):
        """ set new PHI """
        self.__phi = new_phi

    @property
    def phi2(self):
        """ last Euler angle """
        return self.__phi2

    @phi2.setter
    def phi2(self, new_phi2):
        """ set new phi_2 """
        self.__phi2 = new_phi2

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
        g = np.dot(g_phi2, np.dot(g_phi, g_phi1))  # the total orientation matrix is g = g_phi1 * g_phi * g_phi1
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

    #TODO: add conversion to Rodrigues vectors, quaternion


class RotationMatrix(object):
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

    @rotation_matrix.setter
    def rotation_matrix(self, new_r):
        """ modifier for rotation matrix"""
        self.__r = new_r

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


def debug():
    """ Module debugging """
    print "Module debug begins:"


if __name__ == "__main__":
    debug()
