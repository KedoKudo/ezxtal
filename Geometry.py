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
import numpy.linalg as LA


class Point:
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
        assert isinstance(other, Point)
        flag = False
        if np.absolute(self.x - other.x) < 1e-6:
            if np.absolute(self.y - other.y) < 1e-6:
                if np.absolute(self.z - other.z) < 1e-6:
                    flag = True
        return flag

    def __len__(self):
        return 3

    def distance(self, other):
        """Return the distance to another point"""
        assert isinstance(other, Point)
        distance = (self.x - other.x)**2 + (self.y - other.y)**2
        return np.sqrt(distance)

    def is_online(self, line):
        """Quick test is the point is on the given line"""
        assert isinstance(line, Line)
        return line.containsPoint(self)


class Point2D:
    """Point in Plane"""
    def __init__(self, x, y):
        self._coord = [x, y]

    @property
    def x(self):
        """Coordinate x for a 2D point"""
        return self._coord[0]

    @x.setter
    def x(self, val):
        self._coord[0] = val

    @property
    def y(self):
        """Coordinate y for a 2D point"""
        return self._coord[1]

    @y.setter
    def y(self, val):
        self._coord[1] = val

    @property
    def coord(self):
        """Coordinate for a 2D point"""
        return self._coord

    @coord.setter
    def coord(self, val):
        self._coord = val

    def __eq__(self, other):
        assert isinstance(other, Point2D)
        return (self.x == other.x) and (self.y == other.y)

    def __str__(self):
        """string representation of 2D point"""
        return "(" + str(self._coord[1: -1]) + ")"

    def __len__(self):
        return 2

    def __add__(self, other):
        assert isinstance(other, Point2D)
        return Point2D(self.x + other.x, self.y + other.y)

    def distance(self, other):
        assert isinstance(other, Point2D)
        distance = (self.x - other.x)**2 + (self.y - other.y)**2
        return np.sqrt(distance)

    def is_online(self, line):
        """
        quick test to see if a point is on the given line
        """
        assert isinstance(line, Line2D)
        return line.containsPoint(self)


class Line:
    """
    Line in 3D
    """
    def __init__(self, ptStart, ptEnd):
        """
        :param ptStart: A Point object for the starting point
        :param ptEnd: A Point object for the end point
        """
        assert isinstance(ptStart, Point)
        assert isinstance(ptEnd, Point)
        if ptStart == ptEnd:
            print "Need two point to define a line"
            sys.exit(-1)
        else:
            self.ptStart = Point(ptStart.x, ptStart.y, ptStart.z)
            self.ptEnd = Point(ptEnd.x, ptEnd.y, ptEnd.z)

    def __str__(self):
        """
        formatted output for Line object
        """
        outString = "3D Line Object:\n"
        outString += "Starting at ({}, {}, {})\n".format(self.ptStart.x,
                                                         self.ptStart.y,
                                                         self.ptStart.z)
        outString += "End at ({}, {}, {})\n".format(self.ptEnd.x,
                                                    self.ptEnd.y,
                                                    self.ptEnd.z)
        temp_direc = self.getDirection()
        outString += "Direction: <{}, {}, {}>\n".format(temp_direc[0],
                                                        temp_direc[1],
                                                        temp_direc[2])
        return outString

    def __eq__(self, other):
        """
        test if the same line
        """
        assert isinstance(other, Line)
        flag = False
        if self.ptStart == other.ptStart:
            if self.ptEnd == other.ptEnd:
                flag = True
        return flag

    def getDirection(self):
        """
        return a tuple of unit vector denoting line direction
        """
        temp_vector = [self.ptEnd.x - self.ptStart.x,
                       self.ptEnd.y - self.ptStart.y,
                       self.ptEnd.z - self.ptStart.z]
        direc = [float(item)/LA.norm(temp_vector) for item in temp_vector]
        return direc

    def getLength(self):
        """
        return the length of the Line object
        """
        temp_vector = [self.ptEnd.x - self.ptStart.x,
                       self.ptEnd.y - self.ptStart.y,
                       self.ptEnd.z - self.ptStart.z]
        return LA.norm(temp_vector)

    def isParallel(self, other):
        """
        test if two Line objects are parallel in space
        """
        assert isinstance(other, Line)
        flag = False
        direc_0 = self.getDirection()
        direc_1 = other.getDirection()
        if 1 - np.absolute(np.dot(direc_0, direc_1)) < 1e-6:
            flag = True
        return flag

    def containsPoint(self, point):
        """
        test is a point is on line
        """
        assert isinstance(point, Point)
        flag = False
        if point == self.ptStart:
            flag = True
        elif point == self.ptEnd:
            flag = True
        else:
            temp_line = Line(self.ptStart, point)
            if self.isParallel(temp_line):
                minX = min(self.ptStart.x, self.ptEnd.x)
                maxX = max(self.ptStart.x, self.ptEnd.x)
                minY = min(self.ptStart.y, self.ptEnd.y)
                maxY = max(self.ptStart.y, self.ptEnd.y)
                minZ = min(self.ptStart.z, self.ptEnd.z)
                maxZ = max(self.ptStart.z, self.ptEnd.z)
                if minX <= point.x <= maxX:
                    if minY <= point.y <= maxY:
                        if minZ <= point.z <= maxZ:
                            flag = True
        return flag

    def isCoplanar(self, other):
        """
        quick test if two lines are in the same plane
        """
        assert isinstance(other, Line)
        flag = False
        if self.isParallel(other):
            # parallel planes are always coplanar
            flag = True
        else:
            # non-parallel case: skew or intercept
            normal = np.cross(self.getDirection(), other.getDirection())
            temp_line = Line(self.ptStart, other.ptStart)
            test = np.dot(normal, temp_line.getDirection())
            if np.absolute(test) < 1e-6:
                # 90 degree means coplanar
                flag = True
        return flag

    def getDist2Line(self, other):
        """
        return the distance between two line is two lines are skew/parallel
        """
        assert isinstance(other, Line)
        if self.isParallel(other):
            # if two line are parallel to each other
            if self.ptStart != other.ptStart:
                temp_line = Line(self.ptStart, other.ptStart)
            else:
                temp_line = Line(self.ptStart, other.ptEnd)
            normal = np.cross(temp_line.getDirection(), self.getDirection())
            normal = [float(item)/LA.norm(normal) for item in normal]
            vDist = np.cross(normal, self.getDirection())
            vDist = [float(item)/LA.norm(vDist) for item in vDist]  # unit vector along distance direction
            distance = temp_line.getLength() * np.dot(temp_line.getDirection(), vDist)
        elif self.isSkewedFrom(other):
            # two line skewed
            normal = np.cross(self.getDirection(), other.getDirection())
            normal = [float(item)/LA.norm(normal) for item in normal]
            temp_line = Line(self.ptStart, other.ptStart)
            distance = temp_line.getLength() * np.dot(temp_line.getDirection(), normal)
        else:
            # two line intercept
            distance = 0.0
        return np.absolute(distance)

    def isSkewedFrom(self, other):
        """
        quick test to see if two lines are skew from each other
        """
        assert isinstance(other, Line)
        flag = not self.isCoplanar(other)
        return flag

    def getIntercept(self, other):
        """
        return the intercept point by the other line
        """
        assert isinstance(other, Line)
        if self.isParallel(other):
            # parallel lines do not intercept
            intercept = None
        elif self.containsPoint(other.ptStart):
            # the intercept is the start point
            intercept = other.ptStart
        elif self.containsPoint(other.ptEnd):
            # the intercept is the end point
            intercept = other.ptEnd
        elif self.isSkewedFrom(other):
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
            results = LA.solve(Matrix, Vector)
            x = A.x + (B.x - A.x) * results[0]
            y = A.y + (B.y - A.y) * results[0]
            z = A.z + (B.z - A.z) * results[0]
            test = Point(float(x), float(y), float(z))
            # make sure the intercept point is on both lines
            if self.containsPoint(test) & other.containsPoint(test):
                intercept = test
            else:
                # the intercept point is beyond two line
                intercept = None
        return intercept

    def isIntercepted(self, other):
        """
        quick test whether intercepted by another line
        """
        return self.getIntercept(other) is not None


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
        direc = [float(item)/LA.norm(temp_direc) for item in temp_direc]
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
            results = LA.solve(Matrix, vector)
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

    def containsPoint(self, point):
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
            if edge.containsPoint(point):
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
            if edge.isIntercepted(testLine):
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
