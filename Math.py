#!/usr/bin/env python

__author__ = "KenZ"

# Developer Note:
# This is a simple module that contains some very useful math functions regarding calculation


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
    if abs(sin_val) <= 1e-6:
        theta = 0.0 if cos_val > 0.0 else np.pi  # special case: x-axis
    elif sin_val > 1e-6:
        theta = np.arccos(cos_val)  # 1 & 2 Quadrant
    else:
        theta = 2 * np.pi - np.arccos(cos_val)  # 3 & 4 Quadrant
    return theta


def delta(i, j):
    """ return 1 if i,j are different, 0 otherwise """
    return 0.0 if np.absolute(i - j) < 1e-6 else 1.0


def levi_civita(i, j, k):
    """ return the output of Levi-Civita """
    return int((i - j) * (j - k) * (k - i) / 2)  # source: https://gist.github.com/Juanlu001/2689795


def bravis_miller_2_cartesian(index4, c_over_a=1.58):
    """ convert Bravis-Miller indices to standard Cartesian Coordinates """
    # a 3x4 matrix that will convert both slip direction and plane normal from Bravis-Miller to Standard Miller indices
    miller2cartesian = np.array([[1, 0, 0, 0],
                                 [1 / np.sqrt(3), 2 / np.sqrt(3), 0, 0],
                                 [0, 0, 0, 1 / c_over_a]])
    cartesian = np.dot(miller2cartesian, index4)
    return cartesian / np.linalg.norm(cartesian)


def nelder_mead(dict_vtx, o_func, check_vtx,
                alpha=1, gamma=2, rho=-0.5, sigma=0.5):
    """
    @description: Nelder-Mead method for one iteration
    @parameter: dict_vtx = {f(x_n): vtx_n} where vtx_n is one vertex (INPUT);
                o_func: objective function to evaluate each vertex (REQUIRED);
                check_vtx: function for validating each vertex (REQUIRED);
                alpha = 1  # reflection coefficient;
                gamma = 2  # expansion coefficient;
                rho = -0.5  # contraction coefficient;
                sigma = 0.5  # shrink coefficient
    @reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
    """
    wk_vtx = dict_vtx.copy()  # always working on a copy
    tmp_fs = sorted(wk_vtx.keys())  # sorted objective values
    f_0 = tmp_fs[0]  # best vertex value
    f_nn = tmp_fs[-2]  # 2nd worst value
    f_n = tmp_fs[-1]  # worst vertex value
    vtx_0 = np.array(wk_vtx[f_0])  # best vertex
    vtx_n = np.array(wk_vtx.pop(f_n))  # remove the worst vertex from wk_vtx
    # calculate center of gravity (x_o) for x_i (i=0:n-1)
    tmp = np.array([0.0] * len(vtx_0))
    for key in wk_vtx:
        tmp += np.array(wk_vtx[key])
    vtx_o = tmp / len(wk_vtx)
    # calculate reflection
    vtx_r = vtx_o + alpha * (vtx_o - vtx_n)
    vtx_r = check_vtx(vtx_r)  # check bounds
    # evaluate the reflection vertex
    f_r = o_func(vtx_r)
    # now walk through all possible cases
    if f_0 <= f_r < f_nn:
        wk_vtx[f_r] = vtx_r  # replace the worst vertex with reflection
        return wk_vtx
    elif f_r < f_0:
        # compute expanded vertex
        vtx_e = vtx_o + gamma * (vtx_o - vtx_n)
        vtx_e = check_vtx(vtx_e)  # force check bounds
        f_e = o_func(vtx_e)
        if f_e < f_r:
            wk_vtx[f_e] = vtx_e  # replace the worst vertex with expanded
        else:
            wk_vtx[f_r] = vtx_r  # replace the worst vertex with reflection
        return wk_vtx
    elif f_r >= f_nn:
        # compute contracted point
        vtx_c = vtx_o + rho * (vtx_o - vtx_n)
        vtx_c = check_vtx(vtx_c)  # force check newly computed vertex
        f_c = o_func(vtx_c)
        if f_c < f_n:
            wk_vtx[f_c] = vtx_c  # replace the worst vertex with contraction
        else:
            # rare case that contracting away from the largest point increases
            # f, something that cannot happen sufficiently close to a
            # non-singular minimum. In that case we contract towards the
            # lowest point in the expectation of finding a simpler landscape.
            tmp_vtxs = wk_vtx
            wk_vtx = {f_0: vtx_0}
            tmp_vtxs.pop(f_0)  # remove the best case from update queue
            tmp_vtxs[f_n] = vtx_n  # put the worst vertex back in queue
            for key in tmp_vtxs:
                tmp_vtx = np.array(tmp_vtxs[key])
                new_vtx = vtx_0 + sigma * (tmp_vtx - vtx_0)
                new_vtx = check_vtx(new_vtx)  # force check new vertex
                new_f = o_func(new_vtx)
                wk_vtx[new_f] = new_vtx
                # after all update finished
        return wk_vtx


def meshgridT(*arrs):
    """code inspired by http://stackoverflow.com/questions/1827489/numpy-meshgrid-in-3d"""
    arrs = tuple(reversed(arrs))
    arrs = tuple(arrs)
    lens = np.array(map(len, arrs))
    dim = len(arrs)
    ans = []
    for i, arr in enumerate(arrs):
        slc = np.ones(dim, 'i')
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)

        ans.insert(0, arr2)
    return tuple(ans)


def debug():
    """
    For debug purpose
    """
    print "Debug begins"
    # check for find_angle
    for i in range(0, 1000):
        theta = 2 * np.pi / 1e3 * i
        error = theta - find_angle(np.sin(theta), np.cos(theta))
        if error > 1e-10:
            print "{}".format(error)


if __name__ == "__main__":
    debug()