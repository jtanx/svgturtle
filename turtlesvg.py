'''
Renders an SVG document on a Turtle canvas.
'''

from __future__ import print_function
import xml.etree.ElementTree as ET
import copy
import math
import re

NUMBER_REGEX = re.compile(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?')


class Point(object):
    ''' Generic point structure '''

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __repr__(self):
        return 'Point({x},{y})'.format(**self.__dict__)

    def __str__(self):
        return '<{x}, {y}>'.format(**self.__dict__)


class TransformationMatrix(object):
    '''
    Represents an SVG transformation matrix
    https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform
    '''

    def __init__(self, a=1, b=0, c=0, d=1, e=0, f=0):
        self.a, self.c, self.e, \
            self.b, self.d, self.f = \
            a, c, e, \
            b, d, f

    @staticmethod
    def translate(x, y=0):
        return TransformationMatrix(e=x, f=y)

    @staticmethod
    def scale(x, y=None):
        if y is None:
            y = x
        return TransformationMatrix(a=x, d=y)

    @staticmethod
    def rotate(a, x=None, y=None):
        if (x is None) != (y is None):
            raise ValueError("x and y must be set in tandem")
        elif x is not None:
            return TransformationMatrix.translate(x, y) * \
                TransformationMatrix.rotate(a) * \
                TransformationMatrix.translate(-x, -y)

        a = math.radians(a)
        return TransformationMatrix(a=math.cos(a), b=math.sin(a),
                                    c=-math.sin(a), d=math.cos(a))

    @staticmethod
    def skewX(a):
        return TransformationMatrix(c=math.tan(math.radians(a)))

    @staticmethod
    def skewY(a):
        return TransformationMatrix(b=math.tan(math.radians(a)))

    def __mul__(self, other):
        if isinstance(other, TransformationMatrix):
            a = self.a * other.a + self.c * other.b
            b = self.b * other.a + self.d * other.b
            c = self.a * other.c + self.c * other.d
            d = self.b * other.c + self.d * other.d
            e = self.e + self.a * other.e + self.c * other.f
            f = self.f + self.b * other.e + self.d * other.f
            return TransformationMatrix(a, b, c, d, e, f)
        elif isinstance(other, Point):
            x = self.a * other.x + self.c * other.y + self.e
            y = self.b * other.x + self.d * other.y + self.f
            return Point(x, y)
        else:
            return NotImplemented

    def __repr__(self):
        return 'TransformationMatrix({a},{b},{c},{d},{e},{f})'.format(**self.__dict__)

    def __str__(self):
        return '{a}\t{c}\t{e}\n{b}\t{d}\t{f}\n0\t0\t1'.format(**self.__dict__)


class Style(object):
    def __init__(self, stroke_width=None, stroke=None, fill=None):
        self.stroke_width = stroke_width
        self.stroke = stroke
        self.fill = fill

    def apply(self, t, matrix):
        sw = float(self.stroke_width) if self.stroke_width is not None else 0.
        sw = sw * matrix.a * matrix.a / matrix.d
        print("APPLYING:", self, '(', sw, ')')
        # this is so gross
        fill = self.fill if self.fill and self.fill.lower() != 'none' else '#000000'
        stroke = self.stroke if self.stroke and self.stroke.lower() != 'none' else '#000000'
        # Fixme??
        if sw == 0:
            t.pencolor(fill)
        else:
            t.pencolor(stroke)
        t.pensize(sw)
        t.fillcolor(fill)
        if (self.fill is None) or (self.fill.lower() != 'none'):
            print("FILLING")
            t.begin_fill()
        else:
            t.end_fill()

    @staticmethod
    def convert_colour(strcolour):
        '''Converts rep ie rgb(r,g,b) to #rrggbb'''
        if (strcolour is not None) and 'rgb' in strcolour:
            parts = re.sub('[rgb()]', '', strcolour).split(',')
            strcolour = '#' + "".join("%02x" % (int(num) & 0xFF)
                                      for num in parts)
        if strcolour is not None:
            if strcolour.startswith('#') and len(strcolour) == 4:
                strcolour = '#{0}{0}{1}{1}{2}{2}'.format(*strcolour[1:])
        return strcolour

    def rev_update(self, other):
        ''' Update style with other style; current style takes precedence '''
        if self.stroke_width is None:
            self.stroke_width = other.stroke_width
        self.stroke = self.stroke or other.stroke
        self.fill = self.fill or other.fill

    def parse(self, node):
        fill = self.convert_colour(node.attrib.get('fill', self.fill))
        stroke = self.convert_colour(node.attrib.get('stroke', self.stroke))
        stroke_width = node.attrib.get('stroke-width', self.stroke_width)

        if 'style' in node.attrib:
            for style in node.attrib['style'].split(';'):
                if 'fill:' in style:
                    fill = self.convert_colour(
                        style.replace('fill:', '').strip())
                elif 'stroke-width:' in style:
                    stroke_width = float(style.replace('stroke-width:', ''))
                elif 'stroke:' in style:
                    stroke = self.convert_colour(
                        style.replace('stroke:', '').strip())
        elif stroke_width is None:
            # wtf yo
            if 'stroke' in node.attrib:
                stroke_width = 1.

        self.fill, self.stroke, self.stroke_width = fill, stroke, stroke_width

    def __repr__(self):
        return 'Style({stroke_width}, {stroke}, {fill})'.format(**self.__dict__)


class Renderable(object):
    def __init__(self, node=None, other=None, parent=None, **kwargs):
        if other is not None:
            # Copy constructor
            self.viewport_width = other.viewport_width
            self.viewport_height = other.viewport_height
            self.matrix = copy.deepcopy(other.matrix)
            self.style = copy.deepcopy(other.style)
            self.gid = None
            self.gidmap = other.gidmap
            self.parent = parent
            self.children = [x.__class__(other=x, parent=self)
                             for x in other.children]
        else:
            self.viewport_width = parent.viewport_width if parent else None
            self.viewport_height = parent.viewport_height if parent else None
            self.matrix = TransformationMatrix()
            self.style = Style()
            self.gid = None
            self.gidmap = parent.gidmap if parent else {}
            self.parent = parent
            self.children = []

        if node is not None:
            if self.gid is None and 'id' in node.attrib:
                self.gid = node.attrib['id']
                if self.gid in self.gidmap:
                    raise Exception("Duplicate id")
                self.gidmap[self.gid] = self
            self.try_update_style(node)
            self.try_transform(node)

    def parse_length(self, input, horiz):
        ''' Parses an input length to user units '''
        if input is None:
            return 0
        m = re.match(r'^([+-]?[\d]+(?:\.[\d]*?)?)([a-zA-Z%]+)?$', input)
        if not m:
            raise ValueError("Invalid input: " + input)
        value = float(m.group(1))
        print('parsing', value, m.group(0))
        if m.group(2) is not None:
            unit = m.group(2).lower()
            if unit in ('em', 'ex'):
                raise NotImplementedError("Unit is not supported: " + input)
            elif unit == 'px':
                pass
            elif unit == 'pt':
                value *= 1.25
            elif unit == 'pc':
                value *= 15.
            elif unit == 'mm':
                value *= 3.543307
            elif unit == 'cm':
                value *= 35.43307
            elif unit == 'in':
                value *= 90.
            elif unit == '%':
                # We need to get the values in viewBox coordinates
                parent = self
                while parent.parent is not None:
                    parent = parent.parent
                # Apply viewBox scaling
                if horiz:
                    value *= (self.viewport_width / parent.matrix.a + parent.matrix.e) / 100.
                else:
                    value *= (self.viewport_height / parent.matrix.d + parent.matrix.f) / 100.
            else:
                raise ValueError("Invalid units: " + input)
        return value

    def add_child(self, clazz, *args, **kwargs):
        c = clazz(*args, parent=self, **kwargs)
        self.children.append(c)
        return c

    def get_style_and_matrix(self):
        style = copy.deepcopy(self.style)
        matrix = self.matrix
        parent = self.parent
        while parent is not None:
            style.rev_update(parent.style)
            matrix = parent.matrix * matrix
            parent = parent.parent
        return (style, matrix)

    def try_update_style(self, node):
        self.style.parse(node)
        print("Updated", self, self.style)

    def try_transform(self, node):
        if 'transform' in node.attrib:
            ft = TransformationMatrix()
            trs = re.findall(r'[a-z]+\([^\)]+\)', node.attrib['transform'])
            for tr in trs:
                tr = re.split(r'[\(\)]', tr)
                op = tr[0]
                args = [float(x) for x in NUMBER_REGEX.findall(tr[1])]

                print("TR:", op, args)
                if op == 'matrix':
                    ft *= TransformationMatrix(*args)
                elif op == 'translate':
                    ft *= TransformationMatrix.translate(*args)
                elif op == 'scale':
                    ft *= TransformationMatrix.scale(*args)
                elif op == 'rotate':
                    ft *= TransformationMatrix.rotate(*args)
                elif op == 'skewX':
                    ft *= TransformationMatrix.skewX(*args)
                elif op == 'skewY':
                    ft *= TransformationMatrix.skewY(*args)
                else:
                    raise NotImplementedError(
                        "Unknown transformation: " + node.attrib['transform'])
            self.transform(ft)

    def transform(self, matrix, reverse=False):
        if reverse:
            self.matrix *= matrix
        else:
            self.matrix = matrix * self.matrix

    def render(self, t):
        for child in self.children:
            child.render(t)


class NonRenderable(Renderable):
    def __init__(self, **kwargs):
        super(NonRenderable, self).__init__(**kwargs)

    def render(self, t):
        pass


class Circle(Renderable):
    def __init__(self, node=None, other=None, **kwargs):
        super(Circle, self).__init__(node=node, other=other, **kwargs)
        if other is not None:
            self.centre = copy.deepcopy(other.centre)
            self.radius = other.radius
        else:
            self.centre = Point(float(node.attrib.get('cx', 0.)),
                                float(node.attrib.get('cy', 0.)))
            self.radius = float(node.attrib['r'])

    def render(self, t):
        bez = []
        # lol http://spencermortensen.com/articles/bezier-circle/
        # probably easier just to estimate it directly
        fac = 0.55191502449 * self.radius
        fac = Point(self.centre.x + fac, self.centre.y + fac)
        path = ''
        path += 'M{},{}C{},{},{},{},{},{}'.format(
            self.centre.x, self.centre.y + self.radius,
            fac.x, self.centre.y + self.radius,
            self.centre.x + self.radius, fac.y,
            self.centre.x + self.radius, self.centre.y
        )
        path += 'C{},{},{},{},{},{}'.format(
            self.centre.x + self.radius, -fac.y,
            fac.x, self.centre.y - self.radius,
            self.centre.x, self.centre.y - self.radius,
        )
        path += 'C{},{},{},{},{},{}'.format(
            -fac.x, self.centre.y - self.radius,
            self.centre.x - self.radius, -fac.y,
            self.centre.x - self.radius, -self.centre.y,
        )
        path += 'C{},{},{},{},{},{}z'.format(
            self.centre.x - self.radius, fac.y,
            -fac.x, self.centre.y + self.radius,
            self.centre.x, self.centre.y + self.radius,
        )

        print(path)
        tt = Path(path=path, parent=self)
        tt.render(t)
        super(Circle, self).render(t)


class Polygon(Renderable):
    def __init__(self, node=None, other=None, **kwargs):
        if other is not None:
            self.pts = copy.deepcopy(other.pts)
        elif node is not None and 'points' in node.attrib:
            points = [float(x)
                      for x in NUMBER_REGEX.findall(node.attrib['points'])]
            self.pts = []
            for i in range(1, len(points), 2):
                self.pts.append(Point(points[i - 1], points[i]))
        super(Polygon, self).__init__(node=node, other=other, **kwargs)

    def render(self, t):
        style, matrix = self.get_style_and_matrix()
        style.apply(t, matrix)

        first = True
        for pt in self.pts:
            pt = matrix * pt
            if first:
                t.svg_moveto(pt)
                first = False
            else:
                t.svg_lineto(pt)
        t.svg_lineto(matrix * self.pts[0])
        t.svg_end()

        super(Polygon, self).render(t)


class Rect(Polygon):
    def __init__(self, node=None, other=None, **kwargs):
        super(Rect, self).__init__(node=node, other=other, **kwargs)
        if other is None and node is not None:
            p1 = Point(self.parse_length(node.attrib.get('x'), True),
                       self.parse_length(node.attrib.get('y'), False))
            p2 = p1 + Point(self.parse_length(node.attrib['width'], True),
                            self.parse_length(node.attrib['height'], False))
            p11 = Point(p1.x, p2.y)
            p22 = Point(p2.x, p1.y)
            self.pts = [p1, p11, p2, p22]

    def __str__(self):
        try:
            return 'Rect({p1},{p2})'.format(**self.__dict__)
        except KeyError:
            return super(Rect, self).__str__()


class Path(Renderable):
    def __init__(self, node=None, other=None, path=None, *args, **kwargs):
        super(Path, self).__init__(node=node, other=other, **kwargs)
        if other is not None:
            self.path = other.path
            self.upos = Point()
        else:
            self.path = path or node.attrib['d']
            self.upos = Point()

    def getpoints(self, parts, i, convert=True):
        partslist = []

        if parts[i][1:]:
            partslist.append(float(parts[i][1:]))
        i += 1
        while i < len(parts):
            if not parts[i][0].isalpha():
                partslist += [float(x) for x in NUMBER_REGEX.findall(parts[i])]
            else:
                break
            i += 1

        if convert:
            if len(partslist) % 2:
                raise ValueError("Invalid part list " + partslist)

            points = []
            for j in range(1, len(partslist), 2):
                points.append(Point(partslist[j - 1], partslist[j]))
        else:
            points = partslist

        return (points, i)

    def lineto(self, points, t, relative):
        if len(points) < 1:
            return

        for i in range(len(points)):
            if relative:
                pt = points[i] + self.upos
                print("Lineto(R)({})".format(pt))
                t.svg_lineto(self.umatrix * pt)
                self.upos = pt
            else:
                print("Lineto({})".format(points[i]))
                t.svg_lineto(self.umatrix * points[i])
                self.upos = points[i]

    def moveto(self, points, t, relative):
        if len(points) < 1:
            return None

        pt = points[0]
        if relative:
            pt = points[0] + self.upos

        print("Moveto: ", self.upos, '->', pt, '(', self.umatrix * pt, ')')
        t.svg_moveto(self.umatrix * pt)
        self.upos = pt

        if len(points) > 1:
            print("Move->lineto")
            self.lineto(points[1:], t, relative)
        return pt  # Subpath initial pt

    def cubicbezier(self, points, t, relative):
        if len(points) < 3:
            return None

        print("CubicBezier", points, "start", self.upos)
        precision = 0.05
        for i in range(0, len(points), 3):
            cp = self.upos
            lastcp = points[i + 1]
            if relative:
                print(i, len(points), points)
                points[i] += cp
                points[i + 1] += cp
                points[i + 2] += cp

            j = 0
            while j < 1:
                pt = Point(((1. - j)**3) * cp.x + 3 * ((1. - j)**2) * j * points[i].x +
                           3 * (1. - j) * (j**2) *
                           points[i + 1].x + (j ** 3) * points[i + 2].x,
                           ((1. - j)**3) * cp.y + 3 * ((1. - j)**2) * j * points[i].y +
                           3 * (1. - j) * (j**2) *
                           points[i + 1].y + (j**3) * points[i + 2].y)
                t.svg_lineto(self.umatrix * pt)
                j += precision
            t.svg_lineto(self.umatrix * points[i + 2])
            self.upos = points[i + 2]
        return lastcp

    def render(self, t):
        self.upos = Point()
        initial = self.upos
        parts = [x for x in re.split(r"[\s,]|([a-zA-Z])", self.path) if x]
        i = 0
        lastcp = None
        style, matrix = self.get_style_and_matrix()
        self.umatrix = matrix
        print("PATH MAT", repr(style), repr(self.umatrix))

        t.svg_moveto(matrix * initial)
        style.apply(t, matrix)
        while i < len(parts):
            if not parts[i]:
                i += 1
                continue
            command = parts[i][0]
            j = i

            if command == 'M' or command == 'm':
                points, j = self.getpoints(parts, i)
                initial_new = self.moveto(points, t, command == 'm')
                if initial_new:  # If a valid moveto, change initial pt of subpath
                    initial = initial_new
            elif command == 'z' or command == 'Z':
                j = i + 1
                print("Closepath: ", initial)
                # line back to initial pt of subpath
                self.lineto([initial], t, False)
                # Not technically correct, but turtle can't handle self-intersecting paths properly
                t.svg_end()
                style.apply(t, matrix)
            elif command == 'L' or command == 'l':
                points, j = self.getpoints(parts, i)
                self.lineto(points, t, command == 'l')
            elif command == 'H' or command == 'h':
                points, j = self.getpoints(parts, i, False)
                points = [Point(x, 0 if command == 'h' else self.upos.y)
                          for x in points]
                self.lineto(points, t, command == 'h')
            elif command == 'V' or command == 'v':
                points, j = self.getpoints(parts, i, False)
                points = [Point(0 if command == 'v' else self.upos.x, y)
                          for y in points]
                self.lineto(points, t, command == 'v')
            elif command == 'C' or command == 'c':
                print("CURVE", flush=True)
                points, j = self.getpoints(parts, i)
                lastcp = self.cubicbezier(points, t, command == 'c')
            elif command == 'S' or command == 's':
                points, j = self.getpoints(parts, i)
                print("SMOOTHCURVE", points, flush=True)
                for k in range(1, len(points), 2):
                    if lastcp is None:
                        if command == 's':
                            lastcp = Point()
                        else:
                            lastcp = self.upos
                    else:
                        # FIXME: Not really reflection
                        print("REFLECTION: Before", lastcp)
                        # if command == 's':
                        #    lastcp[0], lastcp[1] = -lastcp[0], -lastcp[1]
                        # else:
                        #    ppx, ppy = t.pos()
                        #    lastcp[0] = 2. * ppx - lastcp[0]
                        #    lastcp[1] = 2. * ppy - lastcp[1]
                        print("REFLECTION: After", lastcp)
                    lastcp = self.cubicbezier([lastcp, points[k-1], points[k]], t, command == 's')
            else:
                print("SUPER WARNING: UNSUPPORTED PATH COMMAND", command)
                j = i + 1

            if lastcp is not None and command.lower() not in ('s', 'c'):
                print("CLEARING CP on", command)
                lastcp = None
            i = j

        print("ENDING PATH")
        t.svg_end()
        super(Path, self).render(t)


class TurtleWrapper(object):
    def __init__(self, t):
        super(TurtleWrapper, self).__init__()
        self.t = t
        self.filling = False
        self.pos = Point(0, 0)
        self.is_down = False
        self.t.up()  # Sigh...

    def begin_fill(self):
        self.filling = True
        return self.t.begin_fill()

    def end_fill(self):
        self.filling = False
        return self.t.end_fill()

    def pencolor(self, *args):
        return self.t.pencolor(*args)

    def fillcolor(self, *args):
        return self.t.fillcolor(*args)

    def pensize(self, width=None):
        return self.t.pensize(width=width)

    def up(self):
        if self.is_down:
            self.is_down = False
            return self.t.up()

    def down(self):
        if not self.is_down:
            self.is_down = True
            return self.t.down()

    def setheading(self, angle):
        return self.t.setheading(angle)

    def setposition(self, x, y=None):
        self.pos.x = x
        if y is not None:
            self.pos.y = y
        return self.t.setposition(x=x, y=y)

    def svg_getpos(self):
        pos = self.t.position()
        assert(abs(self.pos.x - pos[0]) < 1e-6)
        assert(abs(self.pos.y - pos[1]) < 1e-6)
        return self.pos

    def svg_end(self):
        if self.filling:
            self.end_fill()
        self.up()

    def svg_moveto(self, pt, down=False):
        if down:
            self.down()
        else:
            self.up()
        pos = self.svg_getpos()
        vec = pt - pos
        heading = math.atan2(vec.y, vec.x) - math.atan2(0, 1)
        #print("MOVETO:", pos, '->', pt)
        self.setheading(math.degrees(heading))
        self.setposition(pt.x, pt.y)

    def svg_lineto(self, pt):
        self.svg_moveto(pt, down=True)


class TurtleSVG(object):
    def __init__(self, file=None, text=None):
        if file is not None:
            self.root = ET.parse(file).getroot()
        elif text is not None:
            self.root = ET.fromstring(text).getroot()
        else:
            raise ValueError("file or text must be specified")

        # Setup the base renderable
        self.base = Renderable()
        self.base.viewport_width = self.base.parse_length(
            self.root.attrib['width'], True)
        self.base.viewport_height = self.base.parse_length(
            self.root.attrib['height'], False)

        if 'viewBox' in self.root.attrib:
            m = NUMBER_REGEX.findall(self.root.attrib['viewBox'])
            if len(m) != 4:
                raise ValueError("Invalid viewbox: " +
                                 self.root.attrib['viewBox'].strip())
            print('VIEWBOX', m, self.base.viewport_width,
                  self.base.viewport_height)
            self.base.transform(TransformationMatrix.scale(
                self.base.viewport_width / float(m[2]),
                self.base.viewport_height / float(m[3])) *
                TransformationMatrix.translate(-float(m[0]), -float(m[1])))

        if 'preserveAspectRatio' in self.root.attrib:
            if self.root.attrib['preserveAspectRatio'].strip().lower() not in ('', 'none'):
                print("Warning: preserveAspectRatio({}) not supported.".format(
                    self.root.attrib['preserveAspectRatio'].strip()))

        self.parse(self.base, self.root)

    def parse(self, parent, children):
        base_ns = '{http://www.w3.org/2000/svg}'
        for child in children:
            tag = child.tag
            if tag.startswith(base_ns):
                tag = tag[len(base_ns):]
            if tag == 'rect':
                parent.add_child(Rect, node=child)
            elif tag == 'circle':
                parent.add_child(Circle, node=child)
            elif tag == 'polygon':
                parent.add_child(Polygon, node=child)
            elif tag == 'path':
                parent.add_child(Path, node=child)
            elif tag == 'defs':
                grp = parent.add_child(NonRenderable, node=child)
                self.parse(grp, child)
            elif tag == 'g':
                grp = parent.add_child(
                    Renderable, node=child)
                self.parse(grp, child)
            elif tag == 'use':
                gid = child.attrib['{http://www.w3.org/1999/xlink}href'].replace(
                    '#', '')
                if gid in parent.gidmap:
                    to_copy = parent.gidmap[gid]
                    cp = to_copy.__class__(
                        node=child, other=to_copy, parent=parent)
                    x = float(child.attrib.get('x', 0))
                    y = float(child.attrib.get('y', 0))
                    if x or y:
                        # Order matters if there's both a transform and x/y
                        cp.transform(TransformationMatrix.translate(x, y), reverse='transform' in child.attrib)
                    parent.children.append(cp)
                else:
                    raise ValueError("Unknown reference: " + gid)
            else:
                # TBA: clipPath
                # Without clipPath, some things probably look weird
                # Well if clipPath is implemented, implement self-intersection checking for path splitting?
                print("WARNING: IGNORING TAG:", tag)

    def render(self, t, setup=True, width=None):
        if setup:
            rwidth, rheight = self.base.viewport_width, self.base.viewport_height
            if width is not None:
                scale = width / rwidth
                rwidth = width
                rheight *= scale
                self.base.transform(TransformationMatrix.scale(scale))
            t.screen.setup(rwidth, rheight, 0, 0)
            t.screen.reset()
            # This is a bit of a hack; it offsets the 0 position by the size of the turtle
            # Hardcoded to 8 pixels for now...
            t.screen.setworldcoordinates(8, rheight, rwidth, 8)
            # t.speed(1)
            t.speed(0)
        t.showturtle()
        self.base.render(TurtleWrapper(t))
        t.hideturtle()


def demo(file, width=None):
    import turtle
    t = turtle.Turtle()
    a = TurtleSVG(file)
    a.render(t, width=width)
    turtle.mainloop()
