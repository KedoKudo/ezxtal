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


class Polygon2D(object):
    """polygon for 2D analysis"""
    def __init__(self):
        """Initialize a 2D polygon with empty vertices list"""
        self.__vertices = []
        self.__edges = []

    def __str__(self):
        """Formatted output for 2D polygon"""
        return "2D {}-Polygon".format(len(self.__edges))

    @property
    def edges(self):
        self.__update()  # use lazy-evaluation, only update when required
        return self.__edges

    @property
    def vertices(self):
        self.__update()  # use lazy-evaluation, only update when required
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
        if point in self.__vertices:
            print "Duplicate vertex: {}".format(point)  # prevent duplicate vertex
        else:
            self.__vertices.append(point)

    def __update(self):
        """
        This function is called to ensure a reasonable and sorted list of vertices exist for each polygon2D instance
        when a new vertex is added to the collection
        """
        ##
        # Note: This is the major part of this class as it is responsible for a reasonable internal data structure of
        #       the polygon in 2D case. Here all vertices will be sorted and stored in a counter-clock direction so that
        #       the connectivity can be easily obtained.
        #       The average center is used to sort the list and order of efficiency is very bad here
        if len(self.__vertices) < 2:
            # empty polygon or single point. No action need
            self.__edges = []
        elif len(self.__vertices) == 2:
            # a new line can be defined here
            temp_line = Line2D(self.__vertices[0], self.__vertices[1])
            self.__edges = [temp_line]
        else:
            # a polygon can be formed now, use the average center as a reference and try to sort the vertex in the list
            # with a counter-clock wise order
            center = self.center  # get the most updated center point
            ##
            # Note: reorder the vertex w.r.t to the newly inserted vertex, i.e. the newly inserted vertex will serve as
            #       the reference pole here.
            new_list = [self.__vertices[-1]]
            ref_line = Line2D(center, new_list[0])
            vertex_dict = {}
            for vertex in self.__vertices[:-1]:
                temp_line = Line2D(center, vertex)
                angle = temp_line.angle2line(ref_line)
                vertex_dict[angle] = vertex
            for key in sorted(vertex_dict.iterkeys()):
                new_list.append(vertex_dict[key])
            # now that the list is fully sorted, construct the edge list
            self.__vertices = new_list
            self.__edges = []
            for index in range(len(self.__vertices)-1):
                self.__edges.append(Line2D(self.__vertices[index],
                                           self.__vertices[index+1]))
            self.__edges.append(Line2D(self.__vertices[-1], self.__vertices[0]))

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
        """
        quick test if a Point2D instance is inside the polygon.
        use a outside point for testing
        """
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
        for i in range(5):
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
            print "intercept is:".format(intercept)
        ##
        # show results
        plt.show()
        print "next cycle"
        counter -= 1

if __name__ == "__main__":
    debug()