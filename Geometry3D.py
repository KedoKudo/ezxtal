#!/usr/bin/env python

__author__ = "KenZ"


#__Developer Note:
#  Common geometry shape in 3D

##
# TODO:
#   Need further test on the 3D case

import numpy as np
import numpy.linalg as La


class Point(object):
    """Point in 3D space"""
    def __init__(self, x, y, z):
        """Initialize a point with given cartesian coordinates"""
        self._coord = [x, y, z]

    @property
    def x(self):
        """Coordinate x for point"""
        return self._coord[0]

    @x.setter
    def x(self, val):
        self._coord[0] = val

    @property
    def y(self):
        """Coordinate y for point"""
        return self._coord[1]

    @y.setter
    def y(self, val):
        self._coord[1] = val

    @property
    def z(self):
        """Coordinate z for point"""
        return self._coord[2]

    @z.setter
    def z(self, val):
        self._coord[2] = val

    @property
    def coord(self):
        """Coordinates of point"""
        return self._coord

    @coord.setter
    def coord(self, val):
        if len(val) != 3:
            raise ValueError("Need 3 coordinates")
        self._coord = val

    def __str__(self):
        """String representation of Point"""
        return "(" + str(self._coord)[1: -1] + ")"

    def __eq__(self, other):
        if np.absolute(self.x - other.x) < 1e-6:
            if np.absolute(self.y - other.y) < 1e-6:
                if np.absolute(self.z - other.z) < 1e-6:
                    return True
        return False

    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return 3

    def dist2point(self, other):
        """Return the distance to another point"""
        assert isinstance(other, Point)
        distance = (self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2
        return np.sqrt(distance)

    def dist2line(self, line):
        """Return the distance to another line"""
        return line.dist2point(self)

    def on_line(self, line):
        """Quick test is the point is on the given line"""
        assert isinstance(line, Line)
        return line.contain_point(self)

    def in_plane(self, plane):
        """Quick test if a point is in a given plane"""
        return plane.contain_point(self)


class Line(object):
    """Line in 3D space"""
    __slots__ = ["_start", "_end"]  # streamline memory for efficiency

    def __init__(self, pt_start, pt_end):
        """Initialize a line with 2 point in 3D space"""
        if pt_start == pt_end:
            raise ValueError("Need two different points to define a line in space")
        else:
            self._start = pt_start
            self._end = pt_end

    @property
    def start_pt(self):
        """Start point of line"""
        return self._start

    @start_pt.setter
    def start_pt(self, new_start):
        self._start = new_start

    @property
    def end_pt(self):
        """End point of line"""
        return self._end

    @end_pt.setter
    def end_pt(self, new_end):
        self._end = new_end

    @property
    def length(self):
        """Length of line"""
        temp = [self.start_pt.x - self.end_pt.x,
                self.start_pt.y - self.end_pt.y,
                self.start_pt.z - self.end_pt.z]
        result = temp[0]**2 + temp[1]**2 + temp[2]**2
        return np.sqrt(result)

    @property
    def direction(self):
        """Direction of line"""
        temp = [self.end_pt.x - self.start_pt.x,
                self.end_pt.y - self.start_pt.y,
                self.end_pt.z - self.start_pt.z]
        result = [float(item/self.length) for item in temp]
        return result

    def __str__(self):
        """String representation of line object"""
        return str(self.start_pt) + "-->" + str(self.end_pt)

    def __neg__(self):
        """line with opposite direction"""
        return Line(self.end_pt, self.start_pt)

    def __eq__(self, other):
        """Test if the same line"""
        if self.start_pt == other.start_pt:
            if self.end_pt == other.end_pt:
                return True
        return False

    def __ne__(self, other):
        return not self == other

    def contain_point(self, point):
        """Test is a point is on line"""
        if point == self.start_pt:
            return True  # special case of start point
        elif point == self.end_pt:
            return True  # special case of end point
        else:
            line1 = Line(point, self.start_pt)
            line2 = Line(point, self.end_pt)
            if np.dot(line1.direction, line2.direction) + 1 < 1e-4:
                return True  # when point online, the angle between line1 and line2 should be 180
        return False

    def parallel_to(self, other):
        """Test if two Line objects are parallel in space"""
        if 1 - np.absolute(np.dot(self.direction, other.direction)) < 1e-4:
            return True
        return False

    def skewed_from(self, other):
        """Quick test if one line is skewed from the other"""
        if self.parallel_to(other):
            return False
        elif self.contain_point(other.start_pt) or self.contain_point(other.end_pt):
            return False  # intercepted at the end point
        else:
            normal = np.cross(self.direction, other.direction)
            normal = [item/La.norm(normal) for item in normal]
            test_line = Line(self.start_pt, other.start_pt)
            if np.absolute(np.dot(normal, test_line.direction)) < 1e-4:
                return False  # two lines are coplanar
            else:
                return True

    def intercepted_by(self, other):
        """Quick test if one line is intercepted by another"""
        return not self.get_intercept(other) is None

    def get_intercept(self, other):
        """Return the intercept point is exist, or return None"""
        if self.parallel_to(other) or self.skewed_from(other):
            return None
        elif self.contain_point(other.start_pt):
            return other.start_pt
        elif self.contain_point(other.end_pt):
            return other.end_pt
        else:
            pt_a = self.start_pt
            pt_b = self.end_pt
            pt_c = other.start_pt
            pt_d = other.end_pt
            matrix = np.array([[pt_b.x - pt_a.x, pt_c.x - pt_d.x],
                               [pt_b.y - pt_a.y, pt_c.y - pt_d.y],
                               [pt_b.z - pt_a.z, pt_c.z - pt_d.z]])
            vector = np.array([pt_c.x - pt_a.x, pt_c.y - pt_a.y, pt_c.z - pt_a.z])
            co_vector = np.dot(matrix.T, vector)
            co_matrix = np.dot(matrix.T, matrix)  # use least-square to solve a overdetermined situation
            results = La.solve(co_matrix, co_vector)
            temp_pt = Point(pt_a.x + (pt_b.x - pt_a.x)*results[0],
                            pt_a.y + (pt_b.y - pt_a.y)*results[0],
                            pt_a.z + (pt_b.z - pt_a.z)*results[0])
            if self.contain_point(temp_pt) and other.contain_point(temp_pt):
                return temp_pt
            else:
                return None

    def dist2point(self, point):
        """Return the distance to a given point"""
        if self.contain_point(point):
            return 0.0
        else:
            temp_line = Line(point, self.start_pt)
            # find the normal of the plane defined by the point and line
            plane_normal = np.cross(temp_line.direction, self.direction)
            plane_normal = [item/La.norm(plane_normal) for item in plane_normal]
            direction = np.cross(self.direction, plane_normal)
            direction = [item/La.norm(direction) for item in direction]
            result = temp_line.length * np.dot(temp_line.direction, direction)
            return np.absolute(result)

    def dist2line(self, other):
        """Return the distance between two skewed or parallel lines"""
        if self.parallel_to(other):
            if self.contain_point(other.start_pt) or self.contain_point(other.end_pt):
                return 0.0  # two line collide
            else:
                return self.dist2point(other.start_pt)
        elif self.skewed_from(other):
            normal = np.cross(self.direction, other.direction)
            normal = [item/La.norm(normal) for item in normal]
            test_line = Line(self.start_pt, other.start_pt)
            result = test_line.length * np.dot(test_line.direction, normal)
            return np.absolute(result)
        else:
            return 0.0

    def angle2line(self, other):
        """Return angle (in degree) to another line"""
        angle = np.arccos(np.dot(self.direction, other.direction)) * 180 / np.pi
        return angle


class Plane(object):
    """Plane with no shape"""
    def __init__(self, point1, point2, point3):
        """Initialize a plane with 3 points"""
        # test if 3 points are on the same line
        if Line(point1, point2).parallel_to(Line(point2, point3)):
            raise ValueError("3 points are colinear ")
        self._point = [point1, point2, point3]

    @property
    def normal(self):
        """Plane normal"""
        normal = np.cross(Line(self._point[0], self._point[1]).direction,
                          Line(self._point[1], self._point[2]).direction)
        normal = [item/La.norm(normal) for item in normal]
        return normal

    def __str__(self):
        out_string = "{}(x - {}) + {}(y - {}) + {}(z - {}) = 0".format(self.normal[0], self._point[0].x,
                                                                       self.normal[1], self._point[0].y,
                                                                       self.normal[2], self._point[0].z)
        return out_string

    def __eq__(self, other):
        if 1 - np.absolute(np.dot(self.normal, other.normal)) < 1e-4:
            return other.contain_point(self._point[0])
        else:
            return False

    def contain_point(self, point):
        """Quick test to see if a point is in plane"""
        test_val = [point.x - self._point[0].x, point.y - self._point[0].y, point.z - self._point[0].z]
        if np.absolute(np.dot(test_val, self.normal)) < 1e-4:
            return True
        else:
            return False

    def contain_line(self, line):
        """Quick test to see if a line lies in a plane"""
        return self.contain_point(line.start_pt) and self.contain_point(line.end_pt)

    def parallel_to(self, other):
        """Quick test if two planes are parallel to each other"""
        if 1 - np.absolute(np.dot(self.normal, other.normal)) < 1e-4:
            return True
        else:
            return False


def debug():
    """Module debug session"""
    print "Module test begins:"


if __name__ == "__main__":
    debug()
