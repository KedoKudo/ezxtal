#!/usr/bin/env python

__author__ = "KenZ"


#__Developer Note:
#  Define common geometry for model construction, including:
#       Point(3D), Point(2D), Line(3D), Line(2D), Plane, Polygon(2D)

##
# TODO:
#   Point: implement dist2plane()
#   Line:  implement dist2plane()
#          implement angle2plane()


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


class Point2D(Point):
    """Point in Plane, derived from the 3D point class"""
    def __init__(self, x, y):
        super(Point2D, self).__init__(x, y, 0)

    def __len__(self):
        return 2


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
                       self.end_pt.y - self.end_pt.y]
        temp_vector = [item/La.norm(temp_vector) for item in temp_vector]
        return temp_vector

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
            matrix = np.array([[pt_b.x - pt_a.x, pt_c.x - pt_d.x],
                               [pt_b.y - pt_a.y, pt_c.y - pt_d.y]])
            vector = np.array([pt_c.x - pt_a.x, pt_c.y - pt_a.y])
            results = La.solve(matrix, vector)
            temp_pt = Point2D(pt_a.x + (pt_b.x - pt_a.x)*results[0],
                              pt_a.y + (pt_b.y - pt_a.y)*results[0],)
            if self.contain_point(temp_pt) and other.contain_point(temp_pt):
                return temp_pt
            else:
                return None

    def skewed_from(self, other):
        """2D lines do not skew from each other"""
        raise TypeError("2D line do not skew from each other")


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

    def add_vertex(self, point):
        """Add one more vertex to the current Polygon"""
        if point in self.__vertices:
            print "Duplicate vertex: {}".format(point)  # prevent duplicate vertex
        else:
            self.__vertices.append(point)

    def __update(self):
        """This function is called to ensure a reasonable and sorted list of vertices exist for each polygon2D instance
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
            center = self.getCenter()  # get the most updated center point
            ##
            # Note: reorder the vertex w.r.t to the newly inserted vertex, i.e. the newly inserted vertex will serve as
            #       the reference pole here.
            newList = [self.__vertices[-1]]
            refLine = Line2D(center, newList[0])
            vertexDict = {}
            for vertex in self.__vertices[:-1]:
                tempLine = Line2D(center, vertex)
                angle = tempLine.getAngleD(refLine)
                vertexDict[angle] = vertex
            for key in sorted(vertexDict.iterkeys()):
                newList.append(vertexDict[key])
            # now that the list is fully sorted, construct the edge list
            self.__vertices = newList
            self.__edges = []
            for index in range(len(self.__vertices)-1):
                self.__edges.append(Line2D(self.__vertices[index],
                                           self.__vertices[index+1]))
            self.__edges.append(Line2D(self.__vertices[-1], self.__vertices[0]))

    def getShortest(self):
        """
        return the shortest distance between the center and vertices
        """
        center = self.getCenter()
        dist = Line2D(center, self.__vertices[-1]).getLength()
        for vertex in self.__vertices[:-1]:
            temp = Line2D(center, vertex).getLength()
            if temp < dist:
                dist = temp
        return dist

    def containsPoint(self, point, rayOrigin=None):
        """
        quick test if a Point2D instance is inside the polygon.
        use a outside point for testing
        """
        assert isinstance(point, Point2D)
        ##
        # First test if the point happens to be on the edges
        for edge in self.getEdges():
            if edge.contain_point(point):
                return True
        ##
        # now start with other settings
        if rayOrigin is None:
            center = self.getCenter()
            temp_x = center.x + 10 * (self.__vertices[-1].x - center.x)
            temp_y = center.y + 10 * (self.__vertices[-1].y - center.y)
            testPoint = Point2D(temp_x, temp_y)
            testLine = Line2D(testPoint, point)
        else:
            assert isinstance(rayOrigin, Point2D)
            testLine = Line2D(rayOrigin, point)
        count = 0
        for edge in self.getEdges():
            if edge.is_intercepted(testLine):
                count += 1
        if count % 2 == 0:
            return False
        else:
            return True

    def getCenter(self):
        """
        return a Point2D class of the geometrical center of the current polygon
        """
        center_x = 0.0
        center_y = 0.0
        for vertex in self.__vertices:
            center_x += float(vertex.x)
            center_y += float(vertex.y)
        center_x /= len(self.__vertices)
        center_y /= len(self.__vertices)
        return Point2D(center_x, center_y)


def Debug():
    """Module debug session"""
    print "Module test begins:"

if __name__ == "__main__":
    Debug()
