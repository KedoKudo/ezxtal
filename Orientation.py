#!/usr/bin/env python

__author__ = "KenZ"

#__Developer Note:
#   A set of class used for describing orientation in space

import numpy as np
import numpy.linalg as LA
import cmath
from Math import find_angle


class EulerAngle:
    """
    Bunge Euler Angle (intrinsic rotation: z->x'->z'')
    """
    def __init__(self, phi1, PHI, phi2):
        """
        store the Euler angle in radians
        """
        deg2rad = np.pi / 180
        self.phi1 = phi1 * deg2rad
        self.PHI = PHI * deg2rad
        self.phi2 = phi2 * deg2rad

    def __eq__(self, other):
        """
        crystal symmetry is not taken into consideration at this point
        """
        assert isinstance(other, EulerAngle)
        flag = False
        # do calculation here, Bunge Euler angle is not 1-on-1 mapping,
        # have to compare the Rotation matrix instead. No symmetry is
        # considered here as the crystal symmetry will have a huge effect
        # and should be considered in the Crystal module
        test = np.dot(self.getRotationAxis(), other.getRotationAxis())
        if 1 - np.absolute(test) < 1e-6:
            # the rotation axis is the same, now considering rotation angle
            test_ang = self.getAngleAroundRotationAxis() - other.getAngleAroundRotationAxis()
            if np.absolute(test_ang) < 1e-6:
                flag = True
        return flag

    def __str__(self):
        """
        output Euler angles using degrees
        """
        rad2deg = 180 / np.pi
        outString = "Bunge Euler angle: [{:.2f}, {:.2f}, {:.2f}]".format(self.phi1 * rad2deg,
                                                                         self.PHI * rad2deg,
                                                                         self.phi2 * rad2deg)
        return outString

    def __len__(self):
        return 3

    def getEulerAngle(self):
        return [self.phi1, self.PHI, self.phi2]

    def getRotationMatrix(self):
        """
        The total rotation matrix:
            R = R_z(phi2)*R_x(PHI)*R_z(phi1)
        """
        R_phi1 = np.array([[np.cos(self.phi1), -np.sin(self.phi1), 0.0],
                           [np.sin(self.phi1), np.cos(self.phi1), 0.0],
                           [0.0, 0.0, 1.0]])
        R_PHI = np.array([[1.0, 0.0, 0.0],
                          [0.0, np.cos(self.PHI), -np.sin(self.PHI)],
                          [0.0, np.sin(self.PHI), np.cos(self.PHI)]])
        R_phi2 = np.array([[np.cos(self.phi2), -np.sin(self.phi2), 0.0],
                           [np.sin(self.phi2), np.cos(self.phi2), 0.0],
                           [0.0, 0.0, 1.0]])
        R = np.dot(R_phi1, np.dot(R_PHI, R_phi2))
        # do calculation here
        return R

    def getBungeMatrix(self):
        """
        return the Bunge Matrix related to phi1, PHI, and phi2
        """
        return self.getRotationMatrix().T

    def getRotationAxis(self):
        rotM = RotationMatrix(self.getRotationMatrix())
        return rotM.getRotationAxis()

    def getQuaternion(self):
        """
        convert the Euler Angle to a Quaternion representation
        """
        pass

    def getAngleAroundRotationAxis(self):
        """
        Get rotation angle around the rotation axis w.r.t the result in
        self.getRotationAxis()
        """
        rotM = RotationMatrix(self.getRotationMatrix())
        return rotM.getAngleAroundRotationAxis()


class RotationMatrix:
    """
    represent the orientation using standard rotation matrix
    """
    def __init__(self, rotM):
        """
        initialize rotation matrix with a 3x3 numpy array
        """
        self.rotationMatrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                self.rotationMatrix[i, j] = rotM[i, j]

    def __str__(self):
        outString = "Rotate Around"
        return outString

    def getRotationAxis(self):
        values, axis = LA.eig(self.rotationMatrix)
        axis = axis.T
        # The rotation axis can only be found by analyze the e^{-i\theta} component.
        # For counter clockwise rotation by theta, the rotation axis is the cross
        # product of v_r and v_i from e^{-i\theta}
        rotAxis = "ERROR"
        for index in range(3):
            if values[index].imag < 0:
                v_r = [item.real for item in axis[index]]
                v_i = [item.imag for item in axis[index]]
                rotAxis = np.cross(v_r, v_i)
                rotAxis /= LA.norm(rotAxis)
        # for non-rotation cases
        if values[0] == values[1] == values[2] == 1:
            rotAxis = np.array([1, 0, 0])
        return rotAxis

    def getAngleAroundRotationAxis(self):
        """
        Get rotation angle around the rotation axis w.r.t the result in
        self.getRotationAxis()
        """
        values, axis = LA.eig(self.rotationMatrix)
        # if it returns a rotation angle that is greater than 360, then something is wrong
        rotAngle = "ERROR"
        for index in range(3):
            if np.absolute(values[index] - 1) > 1e-6:
                rotAngle = np.absolute(cmath.phase(values[index]) * 180.0 / np.pi)
                break
        if rotAngle == "ERROR":
            rotAngle = 0.0
        return rotAngle

    def getEulerAngle(self):
        """
        return one set of Euler Angle w.r.t the rotation matrix
        """
        eulerAngle = [0, 0, 0]
        if np.absolute(self.rotationMatrix[2, 2] - 1.0) < 1e-6:
        #simple one rotation case
            phi1 = find_angle(-self.rotationMatrix[0, 1], self.rotationMatrix[0, 0])
            eulerAngle[0] = phi1 * 180 / np.pi
        else:
            PHI = np.arccos(self.rotationMatrix[2, 2])
            eulerAngle[1] = PHI * 180 / np.pi
            #calculate phi1
            sin_phi1 = self.rotationMatrix[0, 2] / np.sin(PHI)
            cos_phi1 = -self.rotationMatrix[1, 2] / np.sin(PHI)
            phi1 = find_angle(sin_phi1, cos_phi1)
            eulerAngle[0] = phi1 * 180 / np.pi
            #calculate phi2
            sin_phi2 = self.rotationMatrix[2, 0] / np.sin(PHI)
            cos_phi2 = self.rotationMatrix[2, 1] / np.sin(PHI)
            phi2 = find_angle(sin_phi2, cos_phi2)
            eulerAngle[2] = phi2 * 180 / np.pi
        return eulerAngle


class Quaternion:
    """
    represent the orientation in terms of Quaternion
    """
    def __init__(self, w, x, y, z):
        """
        """
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        pass

    def __str__(self):
        """
        """
        outString = ""
        return outString

    def __eq__(self, other):
        """
        """
        assert isinstance(other, Quaternion)
        # do calculation here
        pass

    def __len__(self):
        return 4

    def setFromEulerAngle(self, eulerAngle):
        """
        set Quaternion using Euler Angle
        """
        pass

    def getEulerAngle(self):
        """
        get one set of Euler Angle from current Quaternion
        """
        pass


def Debug():
    """
    Module debugging
    """
    print "Module debug begins:"
    print RotationMatrix(EulerAngle(70, 10, 9).getRotationMatrix()).getEulerAngle()


if __name__ == "__main__":
    Debug()
