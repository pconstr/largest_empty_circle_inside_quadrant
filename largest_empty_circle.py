import numpy
import scipy
import interval
import itertools

from typing import NamedTuple, Generator, Iterable, List, Dict, cast


Point = numpy.ndarray
Vector =  numpy.ndarray
Quadrant = numpy.ndarray


class Circle(NamedTuple):
    c: Point
    r: float


class Halfspace(NamedTuple):
    """
    v points to the outside
    a.x < b
    """
    a: Vector
    b: float

    def strict_contains(self, p):
        return numpy.dot(self.a, p) < self.b


def splitting_halfspace(p1, p2):
    """
    Returns the Halfspace having its boundary halfway  between p1 and p2
    p2 will be outside and p1 inside the halfspace
    """
    a = p2 - p1
    mp = 0.5 * (p1 + p2)
    b = numpy.dot(mp, a)
    return Halfspace(a, b)


def complement(itv):
    "complement of interval"

    chain = itertools.chain(itv, [[interval.inf, None]])

    out = []
    prev = [None, -interval.inf]
    for this in chain:
        if prev[1] != this[0]:
            out.append([prev[1], this[0]])
        prev = this

    return interval.interval(*out)


class Line(NamedTuple):
    "y = ax + b"
    a: float
    b: float


def splitting_line(p0: Point, p1: Point):
    """
    line equidistant from p0 and p1"
    y = ax + b
    """
    d = p1 - p0
    a = - d[0] / d[1]
    mp = 0.5 * (p0 + p1)
    b = mp[1] - a * mp[0]
    return Line(a, b)


def point_in_circle(p: Point, c: Circle):
    "true if point is inside circle"
    
    return (p[0] - c.c[0])**2 + (p[1] - c.c[1])**2 <= c.r**2


def circle_in_quadrant(c: Circle, q: Quadrant):
    "True if circle is inside quadrant"
    
    (x, y), r = c
    if x < r:
        return False
    if x > q[0] - r:
        return False
    if y < r:
        return False
    if y > q[1] - r:
        return False        
    
    return True


def real_roots(p):
    "solve and return only real roots"
    roots = p.roots()
    return roots.real[abs(roots.imag)<1e-6]

def circle_touching_y0_and_two_points(p0: Point, p1: Point):
    if abs(p1[1] - p0[1]) < 1E-6:
        return
    l = splitting_line(p0, p1)
    v0 = p0[0]
    a = l.a
    b = l.b
    v1 = p0[1] - b

    q0 = v0**2 + v1**2 - b**2
    q1 = -2*v0 - 2*v1*a - 2*a*b
    q2 = 1

    polyx = numpy.polynomial.Polynomial([q0, q1, q2])
    for x in real_roots(polyx):
        y = a*x + b
        if y > 0:
            r = y
            yield Circle(numpy.array([x, y]), r)

            
def circle_cornered_x0y0_and_touching_point(p):
    q0 = p[0]**2 + p[1]**2
    q1 = -2*p[0] - 2*p[1]
    q2 = 1
    polyx = numpy.polynomial.Polynomial([q0, q1, q2])
    
    for x in real_roots(polyx):
        if x > 0:
            yield Circle(numpy.array([x, x]), x)


def circles_extending_over_y(q: Quadrant, points) -> Generator[Circle, None, None]:
    """
    0 points touching circle. this can only happen if the circle extends over entire smaller dimension,
    which needs to be y (since a => b)
    this is possible over continuous ranges limited by points, so at the limits it will actually touch a point
    """
    
    r = q[1] / 2
    cy = q[1] / 2
    cy2 = cy*cy
    r2 = r*r
    
    def blocked_cx_range_by_point(p):        
        x, y = p
        hwidth = numpy.sqrt(r2 - (y - cy)**2)
        cxmin = x - hwidth
        cxmax = x + hwidth
        return [cxmin, cxmax]
    
    blocked_cx = interval.interval(*[blocked_cx_range_by_point(p) for p in points])
    allowed_cx =  interval.interval[r, q[0]-r] & complement(blocked_cx)
    
    if allowed_cx:
        # in generate there are infinite, return leftmost
        cx = allowed_cx[0].inf
        c = numpy.array([cx, cy])
        yield Circle(c, r)


def reflect_0(p: Point, v0: float):
    assert p.shape[-1] == 2
    return numpy.concatenate([
        v0 - numpy.expand_dims(
            p[...,0],
            -1
        ),
        numpy.expand_dims(
            p[...,1],
            -1
        )
    ], axis=-1)


def reflect_1(p: Point, v1: float):
    assert p.shape[-1] == 2
    return numpy.concatenate([
        numpy.expand_dims(
            p[...,0],
            -1
        ),
        v1 - numpy.expand_dims(
            p[...,1],
            -1
        )
    ], axis=-1)


def reflect_01(p: Point, q: Quadrant):
    return numpy.array(q) - p


def reflect_0eq1(p: Point):
    return numpy.flip(p, -1)


def point_in_region(p, points, i, vor, ignore_i=None):
    """
    p is inside the region around point i if it's inside all the
    halfspaces between i and the surrounding points
    """
    def inside_halfspace_with(j):
        return splitting_halfspace(points[i],
                                   points[j]).strict_contains(p)
            
    for (ipa, ipb), (vf, vt) in vor.ridge_dict.items():
        if ipa == i:
            if ipb != ignore_i and not inside_halfspace_with(ipb):
                return False
        if ipb == i:
            if ipa != ignore_i and not inside_halfspace_with(ipa):
                return False
    return True    

def circles_touching_1_point(q: Quadrant, points, vor) -> Generator[Circle, None, None]:
    """
    Circle touches one point and two sides of the rectangle
    The center has to be inside the Voroni region of the point
    Needs to be on the correct side of all the halfspaces between that point and the surounding points
    """
    
    for i, p in enumerate(points):
        def in_region(c: Point):
            return point_in_region(c, points, i, vor)

        # against y=0 and x=0
        for c in circle_cornered_x0y0_and_touching_point(p):
            if circle_in_quadrant(c, q):
                if in_region(c.c):
                    yield c

        # against y=q[1] and x=0
        mp = reflect_1(p, q[1])
        for mc in circle_cornered_x0y0_and_touching_point(mp):
            c = Circle(reflect_1(mc.c, q[1]), mc.r)
            if circle_in_quadrant(c, q):
                if in_region(c.c):
                    yield c

        # against x=q[0] and y=0
        mp = reflect_0(p, q[0])
        for mc in circle_cornered_x0y0_and_touching_point(mp):
            c = Circle(reflect_0(mc.c, q[0]), mc.r)
            if circle_in_quadrant(c, q):
                if in_region(c.c):
                    yield c

        # against x=q[0] and y=b
        mp = reflect_01(p, q)
        for mc in circle_cornered_x0y0_and_touching_point(mp):
            c = Circle(reflect_01(mc.c, q), mc.r)
            if circle_in_quadrant(c, q):
                if in_region(c.c):
                    yield c


def circles_touching_2_points(q: Quadrant, points, vor) -> Generator[Circle, None, None]:
    for (ipa, ipb), (vf, vt) in vor.ridge_dict.items():
        pa = points[ipa]
        pb = points[ipb]

        def point_on_ridge(p):
            """
            Check if p is on the ridge between ipa and ipb, assuming it's on the splitting line.
            It needs to be in Voronoi regions of both points sharing the ridge
            """
            return point_in_region(
                p, points, ipa, vor, ignore_i=ipb
            ) and point_in_region(
                p, points, ipb, vor, ignore_i=ipa)

        # against bottom y=0
        for c in circle_touching_y0_and_two_points(pa, pb):
            if circle_in_quadrant(c, q):
                if point_on_ridge(c.c):
                    yield c
                    
        # against top y=q[1]
        mpa = reflect_1(pa, q[1])
        mpb = reflect_1(pb, q[1])
        for mc in circle_touching_y0_and_two_points(mpa, mpb):
            c = Circle(reflect_1(mc.c, q[1]), mc.r)
            if circle_in_quadrant(c, q):
                if point_on_ridge(c.c):
                    yield c

        # against left x=0
        mpa = reflect_0eq1(pa)
        mpb = reflect_0eq1(pb)
        for mc in circle_touching_y0_and_two_points(mpa, mpb):
            c = Circle(reflect_0eq1(mc.c), mc.r)
            if circle_in_quadrant(c, q):
                if point_on_ridge(c.c):
                    yield c

        # against right x=q[0]
        mpa = reflect_0eq1(reflect_0(pa, q[0]))
        mpb = reflect_0eq1(reflect_0(pb, q[0]))
        for mc in circle_touching_y0_and_two_points(mpa, mpb):
            c = Circle(reflect_0(reflect_0eq1(mc.c), q[0]), mc.r)
            if circle_in_quadrant(c, q):
                if point_on_ridge(c.c):
                    yield c


def circles_touching_3_points(q: Quadrant, points, vor) -> Generator[Circle, None, None]:
    """
    3 points touching circle
    The center has to be in one of Voronoi vertices
    """

    all_vertex_points: Dict[int, List[int]] = {}
    for pi, ri in enumerate(vor.point_region):
        point_vertices = vor.regions[ri] 
        for vi in point_vertices:
            if vi != -1:
                vertex_points = all_vertex_points.get(vi, None)
                if vertex_points is None:
                    vertex_points = []
                    all_vertex_points[vi] = vertex_points
                vertex_points.append(pi)

    for vi, vpoints in all_vertex_points.items():
        # take any surrounding point and compute distance, that's the circle radius
        p = points[vpoints[0]]
        v = vor.vertices[vi]
        d = v - p
        r = cast(float, numpy.linalg.norm(d))
        c = Circle(v, r)

        if circle_in_quadrant(c, q):
            yield c
 

def largest_empty_circle_inside_quadrant(q: Quadrant, points, return_candidates=False, return_voronoi=False):
    from time import process_time

    t0 = process_time()
    assert q[0] >= q[1]
    vor = scipy.spatial.Voronoi(points)
    t1 = process_time()
    circles = itertools.chain(
        circles_extending_over_y(q, points),
        circles_touching_1_point(q, points, vor),
        circles_touching_2_points(q, points, vor),
        circles_touching_3_points(q, points, vor)
    )

    result = circles if return_candidates else max(circles, key=lambda c: c.r)
    t2 = process_time()

    if return_voronoi:
        return result, vor
    else:
        return result
