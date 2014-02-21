#!/usr/bin/env python

__author__ = "KenZ"

#__Developer Note:
#   A set of class used for describing orientation in space
#   Assuming that Euler angles are always in degree.


import numpy as np
import numpy.linalg as La
import cmath
from Math import find_angle
from abc import ABCMeta, abstractproperty


##
# MACRO
#
ERROR = 1e-6


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
        pass

    @abstractproperty
    def rotation_matrix(self):
        """ return orientation in the form of rotation matrix """
        pass

    @abstractproperty
    def orientation_matrix(self):
        """ return orientation in the form of orientation matrix """
        pass

    @abstractproperty
    def rotation_angle(self):
        """ return the rotation angle in radians"""
        pass

    @abstractproperty
    def rotation_angled(self):
        """ return the rotation angle in degree """
        pass

    @abstractproperty
    def rotation_axis(self):
        """ return the rotation axis """
        pass

    @abstractproperty
    def quaternion(self):
        """ return teh orientation in the form of Quaternion vector """
        pass


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
        out_str = "[{:.4f}, {:.4f}, {:.4f}]".format(self.__phi1 * rad2deg,
                                                    self.__phi * rad2deg,
                                                    self.__phi2 * rad2deg)
        return out_str

    def __len__(self):
        return 3

    @property
    def euler_angle(self):
        """ Euler angles in degrees """
        rad2deg = 180.0 / np.pi
        return [self.__phi1 * rad2deg, self.__phi * rad2deg, self.__phi2 * rad2deg]

    def set_euler_angle(self, new_angles):
        """ set new Euler angles with vec3"""
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
        #
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
                    g[i, j] = 0.0  # rounding error removed
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
        """ return the rotation angle around the rotation axis in radians"""
        rot_m = RotationMatrix(self.rotation_matrix)
        return rot_m.rotation_angle

    @property
    def rotation_angled(self):
        """ return the rotation angle in degree """
        ang = float(self.rotation_angle) * 180.0 / np.pi
        return ang

    @property
    def quaternion(self):
        """ return the quaternion vector [w, x, y, z] """
        ang = self.rotation_angle
        axs = self.rotation_axis
        tmp_q = [np.cos(ang/2.0), np.sin(ang/2.0)*axs[0], np.sin(ang/2.0)*axs[1], np.sin(ang/2.0)*axs[2]]
        return tmp_q


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
        """ size of the rotation matrix """
        return "3 x 3"

    @property
    def rotation_matrix(self):
        """ accessor for rotation matrix """
        return self.__r

    def set_rotation_matrix(self, new_r):
        """ modifier for rotation matrix with numpy.array((3, 3)) """
        self.__r = new_r

    @property
    def orientation_matrix(self):
        """ return the orientation matrix """
        return self.__r.T

    @property
    def rotation_axis(self):
        values, axis = La.eig(self.__r)
        axis = axis.T
        ang = max([cmath.phase(item) for item in values])
        # Special case
        if abs(ang) < ERROR:
            return [1.0, 0.0, 0.0]  # no rotation at all, pick x as default
        elif abs(ang - np.pi) < ERROR:  # rotate 180 degree
            for i in range(3):
                if abs(1 - values[i]) < ERROR:
                    return np.array(axis[i])
        else:
            for i in range(3):
                if cmath.phase(values[i]) == self.rotation_angle:
                    v_r = [item.real for item in axis[i]]
                    v_i = [-item.imag for item in axis[i]]
                    # The rotation axis can only be found by analyze the e^{-i\theta} component.
                    # For counter clockwise rotation by theta, the rotation axis is the cross
                    # product of v_r and v_i from e^{-i\theta}
                    rot_axs = np.cross(v_r, v_i)
                    return rot_axs / La.norm(rot_axs)

    @property
    def rotation_angle(self):
        """ Get rotation angle around the rotation axis w.r.t the result in self.rotation_axis in radians """
        values, axis = La.eig(self.__r)
        return max([cmath.phase(item) for item in values])  # always choose the positive one (personal preference)

    @property
    def rotation_angled(self):
        """ get rotation angle around the axis in degree """
        ang = float(self.rotation_angle) * 180.0 / np.pi
        return ang

    @property
    def euler_angle(self):
        """ return one set of Euler Angle w.r.t the rotation matrix
            NOTE: Euler Angles are always in degrees """
        angles = [0, 0, 0]
        if np.absolute(self.__r[2, 2] - 1.0) < 1e-6:
        #simple one rotation case
            phi1 = find_angle(-self.__r[0, 1], self.__r[0, 0])
            angles[0] = phi1 * 180.0 / np.pi
        else:
            phi = np.arccos(self.__r[2, 2])
            angles[1] = phi * 180.0 / np.pi
            #calculate __phi1
            sin_phi1 = self.__r[0, 2] / np.sin(phi)
            cos_phi1 = -self.__r[1, 2] / np.sin(phi)
            phi1 = find_angle(sin_phi1, cos_phi1)
            angles[0] = phi1 * 180.0 / np.pi
            #calculate __phi2
            sin_phi2 = self.__r[2, 0] / np.sin(phi)
            cos_phi2 = self.__r[2, 1] / np.sin(phi)
            phi2 = find_angle(sin_phi2, cos_phi2)
            angles[2] = phi2 * 180.0 / np.pi
        return angles

    @property
    def quaternion(self):
        """ return an equivalent quaternion [w, x, y ,z] """
        ang = self.rotation_angle
        axs = self.rotation_axis
        tmp_q = [np.cos(ang/2.0), np.sin(ang/2.0)*axs[0], np.sin(ang/2.0)*axs[1], np.sin(ang/2.0)*axs[2]]
        return tmp_q


class Quaternion(XtalOrientation):
    """ Representing orientation using Quaternion """

    __slots__ = ["__w", "__x", "__y", "__z"]

    def __init__(self, q):
        """ initialize a quaternion with vec4 """
        self.__w = q[0]
        self.__x = q[1]
        self.__y = q[2]
        self.__z = q[3]

    def __str__(self):
        """ formatted output for quaternion """
        w = self.w
        v = self.v
        out_str = "[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(w, v[0], v[1], v[2])
        return out_str

    def __add__(self, other):
        """ quaternions can only operate on its self  """
        if isinstance(other, self.__class__):
            tmp_add = [0, 0, 0, 0]
            for index in range(4):
                tmp_add[index] = self.quaternion[index] + other.quaternion[index]
            return tmp_add  # no normalization here
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'.".format(self.__class__, type(other)))

    def __sub__(self, other):
        """ quaternion can only operate on its self """
        if isinstance(other, self.__class__):
            tmp_sub = [0, 0, 0, 0]
            for index in range(4):
                tmp_sub[index] = self.quaternion[index] - other.quaternion[index]
            return tmp_sub
        else:
            raise TypeError("unsupported operand type(s) for -: '{}' and '{}'.".format(self.__class__, type(other)))

    def __mul__(self, other):
        """ standard multiplication between two quaternions """
        if isinstance(other, self.__class__):
            mul_matrix_self = np.array([[self.__w, -self.__x, -self.__y, -self.__z],
                                        [self.__x,  self.__w, -self.__z,  self.__y],
                                        [self.__y,  self.__z,  self.__w, -self.__x],
                                        [self.__z, -self.__y,  self.__x,  self.__w]])
            return np.array(np.dot(mul_matrix_self, other.quaternion))
        else:
            raise TypeError("unsupported operand type(s) for *: '{}' and '{}'.".format(self.__class__, type(other)))

    @property
    def w(self):
        """ return w in [w, x, y, z] """
        return self.__w

    @property
    def v(self):
        """ return the v """
        return [self.__x, self.__y, self.__z]

    @property
    def conj(self):
        """ return conjugate """
        return [self.__w, -self.__x, -self.__y, -self.__z]

    @property
    def quaternion(self):
        """ return the whole quaternion as a list"""
        return [self.__w, self.__x, self.__y, self.__z]

    @property
    def normalized(self):
        """ return the normalized quaternion """
        return self.quaternion / La.norm(self.quaternion)

    @property
    def euler_angle(self):
        """ return an equivalent Euler Angle """
        tmp_rot = RotationMatrix(self.rotation_matrix)
        return tmp_rot.euler_angle

    @property
    def rotation_matrix(self):
        """ return an equivalent rotation matrix """
        w = self.__w
        x = self.__x
        y = self.__y
        z = self.__z
        tmp_rot = np.array([[1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
                            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]])
        return tmp_rot

    @property
    def orientation_matrix(self):
        """ return an equivalent orientation matrix """
        return self.rotation_matrix.T

    @property
    def rotation_axis(self):
        """ return the rotation axis (unit vector) """
        return self.v / La.norm(self.v)

    @property
    def rotation_angle(self):
        """ return the rotation angle in radians """
        return 2 * np.arccos(self.normalized[0])  # need to use the normalized q

    @property
    def rotation_angled(self):
        """ return the rotation angle in degrees """
        return float(self.rotation_angle) * 180.0 / np.pi

    def is_pure(self):
        """ if w = 0, it's a pure quaternion """
        if np.absolute(self.w) < ERROR:
            return True
        else:
            return False


def average_orientation(orientation_list):
    """ return the average orientation using quaternion (http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf) """
    if not all(isinstance(item, XtalOrientation) for item in orientation_list):
        raise TypeError("Only orientation can be averaged")
    tmp = orientation_list.pop(0).quaternion
    tmp_q = Quaternion(tmp)
    tmp_mtrx = np.outer(tmp_q.quaternion, tmp_q.conj)
    for i in range(len(orientation_list)):
        tmp = orientation_list[i].quaternion
        tmp_q = Quaternion(tmp)
        tmp_mtrx += np.outer(tmp_q.quaternion, tmp_q.conj)
    # take the average
    tmp_mtrx /= len(orientation_list) + 1
    eig, vec = La.eig(tmp_mtrx)
    eig = list(eig)
    # the largest eigenvalue correspond to the average quaternion
    return Quaternion(vec.T[eig.index(max(eig))])