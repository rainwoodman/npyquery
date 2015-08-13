"""
Numpy Database

"""
    
# here are a few examples

def test():
    dtype = numpy.dtype(
        [('id', 'i8'),
         ('v3', ('f4', 3)),
         ('s1', 'f4'),
         ('s2', 'f4'),
        ])

    dtype2 = numpy.dtype(
        [('id2', 'i8'),
         ('s3', 'f4'),
         ('s4', 'f4'),
        ])

    data = numpy.empty(5, dtype=dtype)
    data2 = numpy.empty(5, dtype=dtype2)

    for column in data.dtype.fields:
        data[column].flat[:] = numpy.arange(data[column].size)

    for column in data2.dtype.fields:
        data2[column].flat[:] = numpy.arange(data2[column].size)

    table = NumTable(
                {'id': Primary,
                's1' : Unique | Indexed,
                ('s1', 's2') : Unique | Indexed,
                },
                data=(data, data2)
            )
    print table
    print table.indices[('s1',)].findall(2)
    print table.indices[('s1',)].find(2)
    print table.indices[('s1',)].select(1, 4)
    print table.join(table, ('s1', 's2'))

import numpy

class IndexSpec(int): 
    def __str__(self):
        return repr(self)
    def __repr__(self):
        s = set([])
        true = 0
        for key, value in IndexSpec.__dict__.items():
            if not isinstance(value, IndexSpec): continue
            if self & value != 0:
                s.add(key)
                true |= value

        if true != self:
            s.add('0x%X' % (self ^ true))
        return ' | '.join(s)

    def __or__(self, other):
        return IndexSpec(int(self) | int(other))

Primary = IndexSpec.Primary = IndexSpec(1)
Indexed = IndexSpec.Indexed = IndexSpec(2)
Unique = IndexSpec.Unique = IndexSpec(4)

class NumData(object):
    def __new__(kls, data):
        if isinstance(data, dict):
            self = object.__new__(kls)
            self.base = data
            N = [len(d) for key, d in data.items()]
            for n in N:
                if n != N[0]:
                    raise ValueError("Array length mismatch")
            self.size = N[0]
        elif isinstance(data, numpy.ndarray):
            self = object.__new__(kls)
            self.base = data
            self.size = len(data)

        elif isinstance(data, (tuple, list, set)):
            self = NumData.join([
                NumData(item) for item in data
                ])
        return self

    def dtype(self, key):
        shape = self[key].shape
        base = self[key].dtype
        if len(shape) == 1: 
            return base
        else:
            return numpy.dtype((base, shape[1:]))

    def __len__(self):
        return self.size
        
    def __iter__(self):
        if isinstance(self.base, numpy.ndarray):
            return iter(self.base.dtype.fields)
        else:
            return iter(self.base)

    def __getitem__(self, columns):
        if not isinstance(columns, (tuple, list)):
            return self.base[columns]
        else:
            dtype = [(c, self.dtype(c)) for c in columns]
            data = numpy.empty(len(self), dtype)
            for column in columns:
                data[column][...] = self[column]
        return data

    @classmethod
    def join(kls, list):
        data = dict(
            [(key, item[key]) for item in list for key in item])
        return NumData(data) 

            
class NumTable(object):
    """ A relational data table based on numpy array.


        Parameters
        ----------
        array:  ndarray
            
    """

    def __init__(self, specs, data):

        self.indices = {}

        self.data = NumData(data)

        # find primary
        self.primary = None

        specs = dict([
            (key if isinstance(key, tuple) 
                    else (key,), 
                spec) for key, spec in specs.items()])
            
        for columns, spec in specs.items():
            if spec & Primary:
                if self.primary is not None:
                    raise ValueError(
                "Primary key is specified more than once: %s" % self.primary)
                self.primary = columns

        specs[self.primary] |= Indexed | Unique

        for columns, spec in specs.items():
            if spec & Indexed:
                self.indices[columns] = Index(self.data, columns, spec)

        self.specs = specs

    def join(self, other, columns, mode='raise'):
        """ Join two tables by columns. 
            other must be indexed by columns and unique
        """
        assert columns in other.indices

        i2 = other.indices[columns]
        data = self.data[columns]
        return i2.findeach(data, mode=mode)

    def __repr__(self):
        return ("<NumTable Object at 0x%X>" % id(self)
            +   "\nColumns :\n" 
            +
                    '\n'.join(['  %s : %s' % 
                        (key, str(self.data.dtype(key))) for key in self.data])
            +   "\nIndices:\n"
            +
                    '\n'.join(['  %s : %s' % 
                        (str(key), str(ind.spec)) for key, ind in self.indices.items()])
            ) 

class Index(object):
    def __init__(self, data, columns, spec):
        self.columns = columns
        self.spec = spec
        for column in self.columns:
            if len(data.dtype(column).shape) > 0:
                raise ValueError("Non-scalar columns cannot be indexed")


        if len(data) < numpy.iinfo(numpy.uint32).max:
            itype = numpy.uint32
        else:
            itype = numpy.uint64

        self.data = data[self.columns]

        arg = self.data.argsort(order=columns)
        self.indices = numpy.arange(len(self.data), dtype=itype)

        self.indices = self.indices[arg]
        self.data = self.data[arg]
        if spec & Unique:
            if len(self.data) > 1:
                if (self.data[1:] == self.data[:-1]).any():
                    raise ValueError("Unique column is not unique")

    def _create_foo(self, value):
        if not isinstance(value, numpy.ndarray):
            foo = numpy.empty((), self.data.dtype)
        else:
            foo = numpy.empty(len(value), self.data.dtype)

        for column in self.columns:
            if len(self.columns) > 1:
                foo[column] = value[column]
            else:
                foo[column] = value
        return foo 

    def find(self, value, mode='raise'):
        foo = self._create_foo(value)
        arg = self.data.searchsorted(foo)
        notfound = self.data[arg] != foo 
        if notfound and mode=='raise':
            raise IndexError("Value not found")
        return self.indices[arg]

    def findeach(self, value, mode='raise'):
        foo = self._create_foo(value)
        arg = self.data.searchsorted(foo)
        notfound = self.data[arg] != foo 
        if notfound.any() and mode=='raise':
            raise IndexError("Value not found")
        return self.indices[arg]

    def findall(self, value):
        foo = self._create_foo(value)
        return self.indices[
                self.data.searchsorted(foo, side='left'):
                self.data.searchsorted(foo, side='right')
                ]

    def select(self, left, right):
        left = self._create_foo(left)
        right = self._create_foo(right)
        return self.indices[
                self.data.searchsorted(left, side='left'):
                self.data.searchsorted(right, side='right')
                ]


test()
