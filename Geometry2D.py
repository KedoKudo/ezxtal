#!/usr/bin/env python

__author__ = "KenZ"


#__Developer Note:
#   Common geometry class for 2D shape

from ezxtal.Geometry3D import Point
from ezxtal.Geometry3D import Line
import numpy as np
import numpy.linalg as La


class Point2D(Point):
    """Point in Plane, derived from the 3D point class"""
    def __init__(self, x, y):
        super(Point2D, self).__init__(x, y, 0)

    def __len__(self):
        return 2

    def __str__(self):
        return "({}, {})".format(self.x, self.y)


class Line2D(Line):
    """2D line class for planar analysis"""
    def __init__(self, pt_start, pt_end):
        """Using two 2D point to define a 2D line"""
        assert isinstance(pt_start, Point2D)
        assert isinstance(pt_end, Point2D)
        super(Line2D, self).__init__(pt_start, pt_end)

    def __str__(self):
        """Return formatted string output"""
        out_string = "(" + str(self.start_pt.x) + ", " + str(self.start_pt.y) + ")"
        out_string += "-->"
        out_string += "(" + str(self.end_pt.x) + ", " + str(self.end_pt.y) + ")"
        return out_string

    @property
    def direction(self):
        temp_vector = [self.end_pt.x - self.start_pt.x,
                       self.end_pt.y - self.start_pt.y]
        temp_vector = [item/La.norm(temp_vector) for item in temp_vector]
        return temp_vector

    def parallel_to(self, other):
        vec_1 = self.direction
        vec_2 = other.direction
        if 1 - np.absolute(np.dot(vec_1, vec_2)) < 1e-4:
            return True
        else:
            return False

    @staticmethod
    def skewed_from(self, other):
        """2D lines do not skew from each other"""
        raise TypeError("2D line do not skew from each other")

    def get_intercept(self, other):
        """Return the intercept of two lines"""
        if self.parallel_to(other):
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
            matrix = np.array([[pt_a.y - pt_b.y, pt_b.x - pt_a.x],
                               [pt_c.y - pt_d.y, pt_d.x - pt_c.x]])
            vector = np.array([pt_b.x * pt_a.y - pt_a.x * pt_b.y, pt_d.x * pt_c.y - pt_c.x * pt_d.y])
            results = La.solve(matrix, vector)
            temp_pt = Point2D(results[0], results[1])
            if self.contain_point(temp_pt) and other.contain_point(temp_pt):
                return temp_pt
            else:
                return None

    def get_discrete_pts(self, step=5):
        """ return a list of coordinates discretize the line """
        step_size = int(self.length / float(step)) + 2  # get number of segments for current line
        t = np.linspace(0, 1, step_size)
        pts = []
        # discretizing
        for item in t:
            x = self.start_pt.x + (self.end_pt.x - self.start_pt.x) * item
            y = self.start_pt.y + (self.end_pt.y - self.start_pt.y) * item
            pts.append((x, y))
        return pts

    def get_segments(self, step=5):
        """ return a list of segments that can be directly used by matplotlib """
        pt_list = self.get_discrete_pts(step=step)
        segments = []
        for i in range(len(pt_list) - 1):
            segments.append([pt_list[i], pt_list[i+1]])
        return np.array(segments)


class Polygon2D(object):
    """polygon for 2D analysis"""
    def __init__(self):
        """Initialize a 2D polygon with empty vertices list"""
        self.__vertices = []
        self.__ordered = False

    def __str__(self):
        """Formatted output for 2D polygon"""
        return "2D {}-Polygon".format(len(self.__vertices))

    @property
    def edges(self):
        if not self.__ordered:
            self.__update()  # use lazy-evaluation, only update when needed
        ##
        # compute edge list
        edge_list = []
        for i in range(len(self.__vertices) - 1):
            edge_list.append(Line2D(self.__vertices[i], self.__vertices[i+1]))
        edge_list.append(Line2D(self.__vertices[-1], self.__vertices[0]))
        return edge_list

    @property
    def vertices(self):
        if not self.__ordered:
            self.__update()  # use lazy-evaluation, only update when needed
        return self.__vertices

    @property
    def center(self):
        """return a Point2D class of the geometrical center of the current polygon"""
        center_x = 0.0
        center_y = 0.0
        for vertex in self.__vertices:
            center_x += float(vertex.x)
            center_y += float(vertex.y)
        center_x /= len(self.__vertices)
        center_y /= len(self.__vertices)
        return Point2D(center_x, center_y)

    def add_vertex(self, point):
        """Add one more vertex to the current Polygon"""
        self.__vertices.append(point)
        self.__ordered = False

    def __update(self):
        point_list = []
        for vertex in self.__vertices:
            point_list.append((vertex.x, vertex.y))
        ##
        # build an ordered vertices list use convex_hull method
        self.__vertices = []
        for point in convex_hull(point_list):
            self.__vertices.append(Point2D(point[0], point[1]))
        self.__ordered = True

    def get_shortest(self):
        """return the shortest distance between the center and vertices"""
        center = self.center
        dist = Line2D(center, self.__vertices[-1]).length
        for vertex in self.__vertices[:-1]:
            temp = Line2D(center, vertex).length
            if temp < dist:
                dist = temp
        return dist

    def contains_point(self, point, ray_origin=None):
        """quick test if a Point2D instance is inside the polygon."""
        assert isinstance(point, Point2D)
        ##
        # First test if the point happens to be on the edges
        for edge in self.edges:
            if edge.contain_point(point):
                return True
        ##
        # now start with other settings
        if ray_origin is None:
            center = self.center
            temp_x = center.x + 10 * (self.__vertices[-1].x - center.x)
            temp_y = center.y + 10 * (self.__vertices[-1].y - center.y)
            test_point = Point2D(temp_x, temp_y)
            test_line = Line2D(test_point, point)
        else:
            assert isinstance(ray_origin, Point2D)
            test_line = Line2D(ray_origin, point)
        count = 0
        for edge in self.edges:
            if edge.intercepted_by(test_line):
                count += 1
        if count % 2 == 0:
            return False
        else:
            return True


def convex_hull(point_list):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """
    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(point_list))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower[:-1] + upper[:-1]


def debug():
    """quick unit test for Gemetry2D module"""
    print "Unit test starts"
    from random import randint  # use random points to test class
    import matplotlib.pyplot as plt  # use matplotlib for visualization
    counter = 10
    while counter > 0:
        ##
        # test for polygon2D
        plt.subplot(211)
        test_polygon = Polygon2D()
        for i in range(10):
            temp_pt = Point2D(randint(0, 20), randint(0, 20))
            test_polygon.add_vertex(temp_pt)
        x_list = []
        y_list = []
        for vertex in test_polygon.vertices:
            x_list.append(vertex.x)
            y_list.append(vertex.y)
            plt.plot(vertex.x, vertex.y, "ks")
        x_list.append(x_list[0])
        y_list.append(y_list[0])
        plt.plot(x_list, y_list)
        test_point = Point2D(randint(0, 20), randint(0, 20))
        if test_polygon.contains_point(test_point):
            plt.plot(test_point.x, test_point.y, "r.")
        else:
            plt.plot(test_point.x, test_point.y, "g^")
        test_point = Point2D(10, 10)
        if test_polygon.contains_point(test_point):
            plt.plot(test_point.x, test_point.y, "r.")
        else:
            plt.plot(test_point.x, test_point.y, "g^")
        plt.xlim((-5, 25))
        plt.ylim((-5, 25))
        ##
        # test for line2D
        plt.subplot(212)
        temp_pt1 = Point2D(randint(0, 20), randint(0, 20))
        temp_pt2 = Point2D(randint(0, 20), randint(0, 20))
        temp_pt3 = Point2D(randint(0, 20), randint(0, 20))
        temp_pt4 = Point2D(randint(0, 20), randint(0, 20))
        temp_line1 = Line2D(temp_pt1, temp_pt2)
        temp_line2 = Line2D(temp_pt3, temp_pt4)
        plt.plot([temp_pt1.x, temp_pt2.x], [temp_pt1.y, temp_pt2.y])
        plt.plot([temp_pt3.x, temp_pt4.x], [temp_pt3.y, temp_pt4.y])
        intercept = temp_line1.get_intercept(temp_line2)
        if temp_line1.intercepted_by(temp_line2):
            plt.plot(intercept.x, intercept.y, "ro")
            print "intercept is: {}".format(intercept)
        ##
        # show results
        plt.show()
        print "next cycle"
        counter -= 1

if __name__ == "__main__":
    debug()