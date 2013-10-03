#!/usr/bin/env python

__author__ = "KenZ"


#__Developer Note:
#  Define common geometry for model construction, including:
#       Point(3D), Point(2D), Line(3D), Line(2D), Plane
#       Polygon(3D), Polygon(2D)
#__TODO:
#       Only skeleton for plane and polygon class
#

import sys
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

    def distance(self, other):
        """Return the distance to another point"""
        assert isinstance(other, Point)
        distance = (self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2
        return np.sqrt(distance)

    def is_online(self, line):
        """Quick test is the point is on the given line"""
        assert isinstance(line, Line)
        return line.contain_point(self)


class Point2D(Point):
    """Point in Plane, derived from the 3D point class"""
    def __init__(self, x, y):
        super(Point2D, self).__init__(x, y, 0)

    def __len__(self):
        return 2


class Line:
    """Line in 3D space"""
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

    def is_parallel(self, other):
        """Test if two Line objects are parallel in space"""
        if 1 - np.absolute(np.dot(self.direction, other.direction)) < 1e-4:
            return True
        return False

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

    def is_coplanar(self, other):
        """Quick test if two lines are in the same plane"""
        if self == other:
            return True  # special case where two line are the same
        elif self == -other:
            return True  # special case where two line are mirrored
        elif self.is_parallel(other):
            return True  # Parallel lines are always coplanar
        else:  # either skew or intercept
            if self.contain_point(other.start_pt):
                return True  # two line intercept, thus coplanar
            elif self.contain_point(other.end_pt):
                return True  # two line intercept, thus coplanar
            else:
                normal = np.cross(self.direction, other.direction)
                temp_line = Line(self.start_pt, other.start_pt)
                if np.absolute(normal, temp_line) < 1e-4:
                    return True
        return False

    def dist2line(self, other):
        """Return the distance between two line is two lines are skew/parallel"""
        # NOTE: what will happen if two line intercept outside lines
        if self == other or self == -other:
            return 0.0  # special case where two line are the same/mirror
        elif self.is_parallel(other):
            if self.contain_point(other.start_pt) or self.contain_point(other.end_pt):
                return 0.0  # two lines overlapped
            else:
                temp_line = Line(self.start_pt, other.start_pt)
                plane_normal = np.cross(temp_line.direction, self.direction)
                plane_normal = [item/La.norm(plane_normal) for item in plane_normal]
                common_normal = np.cross(plane_normal, self.direction)
                result = temp_line.length * np.dot(common_normal, temp_line.direction)
                result = np.absolute(result)  # the cos could be negative
                if result < 1e-4:
                    result = min(Line(self.start_pt, other.start_pt).length,
                                 Line(self.end_pt, other.start_pt).length,
                                 Line(self.start_pt, other.end_pt).length,
                                 Line(self.end_pt, other.end_pt).length)  # two continuous segments
                return result
        else:
            normal = np.cross(self.direction, other.direction)
            normal = [item/La.norm(normal) for item in normal]
            temp_line = Line(self.start_pt, other.ptStart)
            result = temp_line.length * np.dot(temp_line.direction, normal)
            result = np.absolute(result)  # the cos could be negative
            return result

    def is_skewed_from(self, other):
        """Quick test to see if two lines are skew from each other"""
        return not self.is_coplanar(other)

    def get_intercept(self, other):
        """Return the intercept point by the other line"""
        # NOTE: current cannot work on the overlapping lines
        if self.is_skewed_from(other):
            return None  # skewed lines do not have intercept
        elif self.contain_point(other.start_pt):
            return other.start_pt
        elif self.contain_point(other.end_pt):
            return other.end_pt
        elif self.is_parallel(other):
            return None  # parallel lines do not intercept
        else:
            pass
        '''
        if self.is_parallel(other):
            # parallel lines do not intercept
            intercept = None
        elif self.contain_point(other.ptStart):
            # the intercept is the start point
            intercept = other.ptStart
        elif self.contain_point(other.ptEnd):
            # the intercept is the end point
            intercept = other.ptEnd
        elif self.is_skewed_from(other):
            # two lines are skewed
            intercept = None
        else:
            # this is not the right way to calculate the intercept, try
            # something else...
            # two line intercept
            A = Point(self.ptStart.x, self.ptStart.y, self.ptStart.z)
            B = Point(self.ptEnd.x, self.ptEnd.y, self.ptEnd.z)
            C = Point(other.ptStart.x, other.ptStart.y, other.ptStart.z)
            D = Point(other.ptEnd.x, other.ptEnd.y, other.ptEnd.z)
            Matrix = np.array([[B.x - A.x, C.x - D.x],
                               [B.y - A.y, C.y - D.y],
                               [B.z - A.z, C.z - D.z]])
            Vector = np.array([C.x - A.x, C.y - A.y, C.z - A.z])
            Vector = np.dot(Matrix.T, Vector)
            Matrix = np.dot(Matrix.T, Matrix)
            results = La.solve(Matrix, Vector)
            x = A.x + (B.x - A.x) * results[0]
            y = A.y + (B.y - A.y) * results[0]
            z = A.z + (B.z - A.z) * results[0]
            test = Point(float(x), float(y), float(z))
            # make sure the intercept point is on both lines
            if self.contain_point(test) & other.contain_point(test):
                intercept = test
            else:
                # the intercept point is beyond two line
                intercept = None
        return intercept
        '''

    def is_intercepted(self, other):
        """Quick test whether intercepted by another line"""
        if self.is_skewed_from(other):
            return False
        else:
            if self.is_parallel(other):
                return False
            else:
                if self.dist2line(other) > 1e-4:

        return self.get_intercept(other) is not None


class Line2D:
    """
    2D line class for planar analysis
    """
    def __init__(self, ptStart, ptEnd):
        """
        using two 2D point to define a 2D line
        """
        assert isinstance(ptStart, Point2D)
        assert isinstance(ptEnd, Point2D)
        if ptStart == ptEnd:
            print "ERROR, need two point to define a line"
            sys.exit(-1)
        else:
            self.ptStart = ptStart
            self.ptEnd = ptEnd

    def __eq__(self, other):
        """
        lines have the same starting and ending point can be consider the same line
        """
        assert isinstance(other, Line2D)
        return self.ptStart == other.ptStart and self.ptEnd == other.ptEnd

    def __str__(self):
        outString = "2D Line:\n"
        outString += "Start at: ({}, {})\n".format(self.ptStart.x,
                                                   self.ptStart.y)
        outString += "End at: ({}, {})\n".format(self.ptEnd.x,
                                                 self.ptEnd.y)
        outString += "Line direction: <{}, {}>".format(self.getDirection()[0],
                                                       self.getDirection()[1])
        return outString

    def getDirection(self):
        """
        return a unit vector defining line direction
        """
        temp_direc = [self.ptEnd.x - self.ptStart.x,
                      self.ptEnd.y - self.ptStart.y]
        direc = [float(item)/La.norm(temp_direc) for item in temp_direc]
        return direc

    def getLength(self):
        """
        return the length of line
        """
        temp_val = (self.ptEnd.x - self.ptStart.x) ** 2
        temp_val += (self.ptEnd.y - self.ptStart.y) ** 2
        return np.sqrt(temp_val)

    def getIntercept(self, other):
        """
        return the intercept point of two 2D lines
        """
        assert isinstance(other, Line2D)
        if self.isParallel(other):
            intercept = None
        else:
            A = self.ptStart
            B = self.ptEnd
            C = other.ptStart
            D = other.ptEnd
            Matrix = np.array([[D.y - C.y, C.x - D.x],
                               [B.y - A.y, A.x - B.x]])
            vector = np.array([[(D.y - C.y) * C.x - (D.x - C.x) * C.y],
                               [(B.y - A.y) * A.x - (B.x - A.x) * A.y]])
            results = La.solve(Matrix, vector)
            test = Point2D(results[0], results[1])
            if self.containsPoint(test) & other.containsPoint(test):
                intercept = test
            else:
                intercept = None
        return intercept

    def isIntercepted(self, other):
        """
        quick check is two lines intercept each other
        """
        assert isinstance(other, Line2D)
        return self.getIntercept(other) is not None

    def isParallel(self, other):
        """
        quick check if two lines are parallel to each other
        """
        assert isinstance(other, Line2D)
        test_val = np.dot(self.getDirection(), other.getDirection())
        return (1 - np.absolute(test_val)) < 1e-6

    def contain_point(self, point):
        """
        quick check if a point lies on the current line
        """
        assert isinstance(point, Point2D)
        maxX = max(self.ptStart.x, self.ptEnd.x)
        minX = min(self.ptStart.x, self.ptEnd.x)
        maxY = max(self.ptStart.y, self.ptEnd.y)
        minY = min(self.ptStart.y, self.ptEnd.y)
        flag = False
        if minX == maxX == point.x:
            if minY <= point.y <= maxY:
                return True
        if minY == maxY == point.y:
            if minX <= point.x <= maxX:
                return True
        if minX <= point.x <= maxX:
            if minY <= point.y <= maxY:
                slope = (self.ptEnd.y - self.ptStart.y) / (self.ptEnd.x - self.ptStart.x)
                test = point.y - self.ptStart.y - slope * (point.x - self.ptStart.x)
                if np.absolute(test) < 1e-6:
                    flag = True
        return flag

    def dist2Line(self, other):
        """
        return the distance between two lines
        """
        assert isinstance(other, Line2D)
        if self.isParallel(other):
            # have non-zero distance
            # the normal direction of <x, y> should be <-y, x>
            normal = [-self.getDirection()[1], self.getDirection()[0]]
            temp_line = Line2D(self.ptStart, other.ptStart)
            distance = temp_line.getLength() * np.dot(normal, temp_line.getDirection())
        else:
            # intercepted lines have 0 distance
            distance = 0
        return distance

    def dist2Point(self, point):
        """
        return the distance between a point and the current line
        """
        # don't have to worry about on line case as the output will just be 0
        assert isinstance(point, Point2D)
        normal = [-self.getDirection()[1], self.getDirection()[0]]
        temp_line = Line2D(self.ptStart, point)
        distance = temp_line.getLength() * np.dot(normal, temp_line.getDirection())
        return distance

    def getAngle(self, other):
        """
        return the angle between two Line2D object.
        bullet proofed ^0^
        """
        ##
        # Note: Here the Line2D is treated as vector, thus the angle between two Line2D objects ranges from 0 to 2pi
        assert isinstance(other, Line2D)
        sin_val = np.cross(other.getDirection(), self.getDirection())
        cos_val = np.dot(other.getDirection(), self.getDirection())
        if sin_val >= 0.0:
            return np.arccos(cos_val)
        elif (sin_val < 0.0) & (cos_val < 0.0):
            return np.pi + np.arctan(sin_val/cos_val)
        elif (sin_val < 0.0) & (cos_val == 0.0):
            return 1.5*np.pi
        else:
            return 2*np.pi + np.arctan(sin_val/cos_val)

    def getAngleD(self, other):
        """
        return the angle between 2 Line2D object in degrees
        """
        assert isinstance(other, Line2D)
        angle = self.getAngle(other)
        return angle * 180 / np.pi


class Plane:
    """
    plane with no shape
    """
    def __init__(self):
        pass

    def __str__(self):
        outString = ""
        return outString

    def __eq__(self, other):
        assert isinstance(other, Plane)
        flag = False
        return flag

    def getNormal(self):
        """
        return the unit vector of the plane normal as a tuple
        """
        normal = None
        return tuple(normal)

    def containPoint(self, point):
        """
        quick test to see if a point is in plane
        """
        assert isinstance(point, Point)
        flag = False
        return flag

    def containLine(self, line):
        """
        quick test to see if a line lies in a plane
        """
        assert isinstance(line, Line)
        flag = False
        return flag

    def isParallel(self, other):
        """
        quick test if two planes are parallel to each other
        """
        assert isinstance(other, Plane)
        pass

    def isIntercepted(self, other):
        """
        quick check if two planes intercept each other
        """
        assert isinstance(other, Plane)
        pass

    def getIntercept(self, other):
        """
        return the intercept line of two planes
        """
        assert isinstance(other, Plane)
        pass


class Polygon:
    """
    several coplanar lines connect to create a polygon in space
    """
    pass


class Polygon2D:
    """
    polygon for 2D analysis
    """
    def __init__(self):
        """
        empty vertex list
        """
        self.__vertices = []
        self.__edges = []

    def __str__(self):
        outString = "Polygon Object:\n"
        outString += "total number of vertices: {}\n".format(len(self.__vertices))
        outString += "total number of edges: {}".format(len(self.__vertices))
        return outString

    def __len__(self):
        """
        return the number of vertices/edges for current polygon
        """
        return len(self.__vertices)

    def addVertex(self, point):
        """
        add one more vertex to the current Polygon
        The vertices list is sorted in a different place
        """
        assert isinstance(point, Point2D)
        # prevent duplicate points
        if point in self.__vertices:
            print "Duplicate vertex: {}".format(point)
            # do not add duplicate vertex
            return 0
        else:
            # whenever a new vertex is added, the shape of the polygon will change, thus the center will move around
            # and the order of the vertices need to be recalculated
            self.__vertices.append(point)
            self.__update()

    def __update(self):
        """
        this function is called to ensure a reasonable and sorted list of vertices exist for each polygon2D instance
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

    def getEdges(self):
        """
        return a list of edges associated with the current polygon2D
        """
        return self.__edges

    def getVertices(self):
        """
        return a list of sorted vertices
        """
        return self.__vertices


def Debug():
    """
    module debug session
    """
    print "Module test begins:"
    import matplotlib.pyplot as plt
    p0 = Point2D(0.0, 0.0)
    p1 = Point2D(1.0, 1.0)
    p2 = Point2D(0.0, 1.0)
    p3 = Point2D(1.0, 0.0)
    p4 = Point2D(1.5, 0.5)
    p5 = Point2D(-1.5, 0.5)
    testPoint = Point2D(0.0, 0.3)
    polygon = Polygon2D()
    polygon.addVertex(p0)
    polygon.addVertex(p1)
    polygon.addVertex(p2)
    polygon.addVertex(p3)
    polygon.addVertex(p4)
    polygon.addVertex(p5)
    rayO = Point2D(-5, -5)
    print "Contains {}: {}".format(testPoint, polygon.containsPoint(testPoint, rayOrigin=rayO))
    for edge in polygon.getEdges():
        plt.plot(testPoint.x, testPoint.y, 'ro')
        plt.plot([edge.ptStart.x, edge.ptEnd.x],
                 [edge.ptStart.y, edge.ptEnd.y],
                 'k-')
    plt.show()

if __name__ == "__main__":
    Debug()
