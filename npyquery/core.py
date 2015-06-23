import numpy

class Expr(object):
    def __init__(self, operator, operands):
        self.column = column
        self.operator = operator
        self.operands = operands

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
    def __repr__(self):
        if len(self.operands == 2):
            a, b = self.operands
            return "(%s %s %s)" % (repr(a), self.operator, repr(b))
        elif len(self.operands == 1):
            [a] = self.operands
            return "%s %s" % (self.operator, repr(a))
        else:
            raise ValueError

class Where(Expr):
    def __init__(self, column):
        Expr.__init__(self, "[]", [column])

print Where("A")

