import numpy

class Node(object):
    def __le__(self, other):
        return Expr("<=", [self, other])
    def __lt__(self, other):
        return Expr("<", [self, other])
    def __eq__(self, other):
        return Expr("==", [self, other])
    def __gt__(self, other):
        return Expr(">", [self, other])
    def __ge__(self, other):
        return Expr(">=", [self, other])
    def __and__(self, other):
        return Expr("&", [self, other])
    def __or__(self, other):
        return Expr("|", [self, other])
    def __xor__(self, other):
        return Expr("^", [self, other])
    def __invert__(self):
        return Expr("!", [self])
    def __pow__(self, other):
        return Expr("**", [self, other])
    def __rpow__(self, other):
        return Expr("**", [other, self])
    def __mul__(self, other):
        return Expr("*", [self, other])
    def __div__(self, other):
        return Expr("/", [self, other])
    def __add__(self, other):
        return Expr("+", [self, other])
    def __neg__(self):
        return Expr("-", [self])
    def __sub__(self, other):
        return Expr("-", [self, other])
    def __mod__(self, other):
        return Expr("%", [self, other])
    def __getitem__(self, index):
        return GetItem(self, index)
    def sin(self):
        return Expr("sin", [self])
    def visit(self, array):
        raise NotImplemented 

def repr_slice(s):
    if not isinstance(s, slice):
        return repr(s)
    if s.start is None and s.stop is None and s.step is None:
        return ':'
    start = "" if s.start is None else repr(s.start)
    stop = "" if s.stop is None else repr(s.stop)
    if s.step is None:
        return "%s:%s" % (start, stop)
    else:
        step = repr(s.step)
    return "%s:%s:%s" % (start, stop, step)

class GetItem(Node):
    def __init__(self, obj, index):
        self.obj = obj
        self.index = index
    def __repr__(self):
        if isinstance(self.index, tuple):
            rslice = ','.join([repr_slice(o) for o in self.index])
        else:
            rslice = repr_slice(self.index)
        return "%s[%s]" % (repr(self.obj), rslice)

    def visit(self, array):
        return self.obj.visit(array)[self.index]

class Literal(Node):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return repr(self.value)
    def visit(self, array):
        return self.value

class Column(Node):
    def __init__(self, column):
        Node.__init__(self)
        self.column = column
    def __repr__(self):
        return "[%s]" % self.column
    def visit(self, array):
        return array[self.column]

opmap = {
    '>=': numpy.greater_equal,
    '<=': numpy.less_equal,
    '>': numpy.greater,
    '<': numpy.less,
    '==': numpy.equal,
    '!=': numpy.not_equal,
    '*': numpy.multiply,
    '/': numpy.divide,
    '+': numpy.add,
    '-': {
        1: numpy.negative,
        2: numpy.subtract,
        },
    '~': numpy.bitwise_not,
    '&': numpy.bitwise_and,
    '|': numpy.bitwise_or,
    '^': numpy.bitwise_xor,
    '**' : numpy.power,
    'sin': numpy.sin,
}

class Expr(Node):
    def __init__(self, operator, operands):
        self.operator = operator
        operands = [
            a if isinstance(a, Node)
            else Literal(a)
            for a in operands
        ]
        self.operands = operands

    def __repr__(self):
        if len(self.operands) == 2:
            a, b = self.operands
            return "(%s %s %s)" % (repr(a), self.operator, repr(b))
        elif len(self.operands) == 1:
            [a] = self.operands
            return "(%s %s)" % (self.operator, repr(a))
        else:
            raise ValueError
    def visit(self, array):
        ops = [a.visit(array) for a in self.operands]
        function = opmap[self.operator]
        if isinstance(function, dict):
            function = function[len(ops)]
        return function(* ops)

    
testdata = numpy.empty(2, dtype=[
            ('A', 'f4'),
            ('B', ('f4', 3))])
testdata['A'][0] = 1
testdata['B'][0] = [0, 1, 2]
testdata['A'][1] = 2
testdata['B'][1] = [2, 3, 4]
expr = (-Column('A') >= 1) & (pow(Column('A'), 1) + 3 < 10) & (Column('B')[:, 1] >= 1)

print expr
print expr.visit(testdata)
for clause in expr.operands:
    print clause.visit(testdata)

print testdata[expr.visit(testdata)]


