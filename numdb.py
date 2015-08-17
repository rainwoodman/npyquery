"""
Numpy Database

"""
    
class TestSuite:
    def setup(self):

        dtype = numpy.dtype(
            [('id', 'i8'),
             ('s', 'f4'),
             ('d', 'f8'),
            ])

        dtype2 = numpy.dtype(
            [('id', 'i8'),
             ('s', 'f4'),
            ])

        data = numpy.empty(5, dtype=dtype)
        data2 = numpy.empty(5, dtype=dtype2)

        for column in data.dtype.fields:
            data[column].flat[:] = numpy.arange(data[column].size)[::-1]

        for column in data2.dtype.fields:
            data2[column].flat[:] = numpy.arange(data2[column].size)[::-1]

        data['d'][:] = numpy.floor(data['d'] * 0.5)
        data2['s'][2] = 10

        self.data = data
        self.data2 = data2

        self.table = NumTable(data=self.data)

        self.table2 = NumTable(data=self.data2)

    def test_new_indices(self):
        table = NumTable(data=self.data, specs=
                {'id' : Unique | Indexed })
        table.indices['id']

    def test_index(self):
        table2 = NumTable(data=self.data2, specs=
                {'s' : Unique | Indexed })

        # these are fast find methods on indexed columns
        # need a way to expose these methods
        i3 = table2.indices['s']

        all = table2.indices['s'].findall(3) 

        first = table2.indices['s'].find(3)

        assert (self.data2['s'][all] == 3).all()
        assert all[0] == first

    def test_select_single_column(self):
        table2 = self.table2

        where = table2['id'] != 3

        # mimic SQL select.
        sel = table2.select('id',
                    where=where)
        assert len(sel.columns()) == 1

        sel2 = table2.select(('id',), where)

        assert len(sel2.columns()) == 1

        sel3 = table2.select(C('id'), where)

        assert len(sel3.columns()) == 1
    
    def test_select_many_columns(self):
        table2 = self.table2

        where= table2['id'] != 3

        # mimic SQL select.
        sel = table2.select(('id', 's'),
                    where=where)
        assert len(sel.columns()) == 2

        sel2 = table2.select(C('id', 's'), where)

        assert len(sel2.columns()) == 2

        sel3 = table2.select(C('id', s2='s'), where)

        assert len(sel3.columns()) == 2
        assert 's2' in sel3
        assert 'id' in sel3

    def test_fancy_select(self):
 
        table2 = self.table2

        sel = table2.select(
                columns=C(
                  'id',  
                  sin_s=(numpy.sin, 's'),
                  newid=(NewColumn, numpy.arange(table2.size))),
                )
        assert 'sin_s' in sel
        assert 'newid' in sel
        assert 'id' in sel

    def test_join_inner(self):

        table = self.table
        table2 = self.table2

        # inner join
        ij = table.join(table2, 
                on='id', 
                other_columns=C(s2='s'),
                mode=Inner)

        assert 'id1' not in ij
        assert 'id' in ij

    def test_join_left(self):

        table = self.table
        table2 = self.table2

        # inner join
        ij = table.join(table2, 
                on='id', 
                other_columns=C(s2='s'),
                mode=Inner)

        assert 'id1' not in ij
        assert 'id' in ij

    def test_groupby(self):

        table = self.table
        groupby = table.groupby('d', 
                {'sum'  : (sum, table['s']),
                 'count'  : (count, table['s']),
                 'max'  : (max, table['s']),
                 'min'  : (min, table['s']),
                 'first'  : (first, table['s']),
                 'last'  : (last, table['s']),
                 'mean' : (avg, table['s']),
                })

        assert groupby.size == len(numpy.unique(table['d']))
        assert numpy.allclose(groupby['mean'] * groupby['count'], groupby['sum'])
        assert numpy.all(groupby['min'] <= groupby['max'])

        # FIXME: test first last min max

import numpy
import warnings

# Constants

def Aggregations():
    """ Aggregation operators. """

    min = minimum = numpy.minimum.reduceat
    max = maximum = numpy.maximum.reduceat
    sum = numpy.add.reduceat

    def avg(array, offset, axis):
        s = sum(array, offset, axis)
        N = count(array, offset, axis)
        return s / (N * 1.0)

    mean = average = avg 

    def count(array, offset, axis):
        """ Count the number of rows returned. 

            FIXME: Is this DISTINCT? """
        N = numpy.empty_like(offset)
        N[:-1] = offset[1:] - offset[:-1]
        N[-1] = len(array) - offset[-1]
        return N

    def var(array, offset, axis):
        """ Variance """
        xbar = mean(array, offset, axis)
        x2bar = mean(array ** 2, offset, axis)
        return (x2bar - xbar ** 2)

    def std(array, offset, axis):
        """ Standard deviation"""
        var = var(array, offset, axis)
        N = count(array, offset, axis)
        return var / N

    def first(array, offset, axis):
        """ First item 
        
            Note: will give wrong results if some chunks
            are of length 0
        """
        return array[offset]

    def last(array, offset, axis):
        """ Last item. 

            Note: will give wrong results if some chunks
            are of length 0
        """
        offset2 = numpy.empty_like(offset)
        offset2[:-1] = offset[1:] - 1
        offset2[-1] = len(array) - 1
        return array[offset2]

    globals().update(locals())

    return locals()

# replace Aggregations with an object, for easy access of the
# aggregators

Aggregations = type('Aggregations', 
                    (object,), 
                    {'__init__' : 
                        lambda self, d: self.__dict__.update(d),
                     '__doc__' : 'Namespace of aggregation functions.'
                    })(Aggregations())

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

Indexed = IndexSpec.Indexed = IndexSpec(2)
Unique  = IndexSpec.Unique  = IndexSpec(4)

class JoinMode(int): 
    """ Modes for the join operation """
    def __str__(self):
        return repr(self)
    def __repr__(self):
        for key, value in JoinMode.__dict__.items():
            if not isinstance(value, JoinMode): continue
            if value == self: return key

Inner = JoinMode.Inner = JoinMode(0)
Left  = JoinMode.Left  = JoinMode(1)
Right = JoinMode.Right = JoinMode(2)
Outer = JoinMode.Outer = JoinMode(3)

def NewColumn(x):
    """ A special function used by select to create a new column. """
    return x

def C(*args, **kwargs):
    """ Build a list of columns from positional and keyword arguments.

        Examples
        --------

        >>> C('HaloID', 'SubHaloID', 'Mass', LogMass=(numpy.log10, 'Mass'))

        Returns
        -------
        dict :
            a dictionary of suitable to be used as input to the columns arguments.
    """
    d = {}
    for w in args:
        d[w] = w

    d.update(kwargs)
    return d
    
class NumData(object):
    """ A collection of ndarrays, with identical length. """
    def __new__(kls, data):
        if isinstance(data, dict):
            self = object.__new__(kls)
            self.base = data
            N = [len(d) for key, d in data.items()]
            for n in N:
                if n != N[0]:
                    raise ValueError("Array length mismatch")
            self.size = N[0]
            return self
        elif isinstance(data, numpy.ndarray):
            d = dict([
                (key, data[key]) for key in data.dtype.fields])
            return NumData(d)            
        elif isinstance(data, NumData):
            return NumData(data.base)
        elif isinstance(data, list):
            d = {}
            for key, value in data:
                if key not in d:
                    d[key] = value
                else:
                    raise KeyError("Column `%s` already exist" % str(key))
            return NumData(d)

        raise TypeError("Unsupported type `%s` for NumData" % str(type(data)))

    def dtype(self, key):
        shape = self[key].shape
        base = self[key].dtype
        if len(shape) == 1: 
            return base
        else:
            return numpy.dtype((base, shape[1:]))

    def __len__(self):
        raise MethodError("len is ill defined. Use .size for number of rows.")
        
    def __iter__(self):
        return iter(self.base)

    def columns(self):
        return self.base.items()

    def __contains__(self, columns):
        if not isinstance(columns, (tuple, list)):
            return columns in self.base
        else:
            for column in columns:
                if column not in self.base:
                    return False
        return True 

    def toarray(self, columns):
        if not isinstance(columns, (tuple, list)):
            if not columns in self.base:
                raise KeyError('Column `%s` not found' % columns)
            return self.base[columns]
        else:
            dtype = [(c, self.dtype(c)) for c in columns]
            data = numpy.empty(self.size, dtype)
            for column in columns:
                data[column][...] = self[column]
        return data
        
    def __getitem__(self, column):
        if not column in self.base:
            raise KeyError('Column `%s` not found' % str(column))
        return self.base[column]
        
    def __setitem__(self, columns, value):
        raise RuntimeError('NumData is immutable. Create a new object with NumData.concatenate.')

    @classmethod
    def concatenate(kls, list):
        data = []
        for item in list:
            for key in item:
                data.append((key, item[key]))
        return NumData(data) 

class NumTable(object):
    """ A relational data table based on numpy array.


        Parameters
        ----------
        data :  dict or ndarray
            the input data

        specs : 
            How the index is built.

        add_id : string
            Append a unique ID column to the data. 
            Parameter is the name of the ID field. 
    """

    def __init__(self, data, specs=None, add_id=None):
        self.indices = {}
        self.specs = {}
        self.data = NumData(data)
        if specs is not None:
            self.update_indices(specs)
        if add_id is not None:
            self.append_columns(
                {add_id : numpy.arange(len(self))})

    def __getitem__(self, columns):
        return self.data.toarray(columns)

    def __len__(self):
        raise MethodError("len is ill defined. Use .size for number of rows.")

    @property
    def size(self):
        return self.data.size

    def __contains__(self, key):
        return key in self.data

    def columns(self):
        return self.data.columns()

    def append_columns(self, data):
        """ Append columns from data.
            
            I hope this is not used very often.

            select ({'column' : (NewColumn, data))
        """
        self.data = NumData.concatenate((self.data, NumData(data)), rename=False)
 
    def update_indices(self, specs):
        specs = dict([
                (ensure_tuple(key), spec) 
                for key, spec in specs.items()])
            
        for columns, spec in specs.items():
            if spec & Indexed:
                self.indices[columns] = Index(self.data, columns, spec)
                # set up alias for single column indices
                if len(columns) == 1:
                    self.indices[columns[0]] = self.indices[columns]

        self.specs.update(specs)
        
    def select(self, columns=None, where=None):
        """ Select * from table where ...

            Parameters
            ----------
            where  : 
                any object valid as ndarray index. Each column will
                be indexed by where.
                
            columns : 
                Columns to select. Constructed with :py:func:`C`.

                - To rename, use a dict of {newname : oldname}.
                - To apply functions, use a dict of {name : (func, oldname)}.
                - To create columns, use a dict of {name : (NewColumn, data)}. 
                  data[where] is applied to the column.
                    
            Returns
            -------
            NumTable

        """
        if columns is None:
            columns = tuple(iter(self.data))

        columns = ensure_dict(columns)        
        if where is None:
            where = Ellipsis
        elif hasattr(where, "__call__"):
            where = where(self)

        data = {}
        for asc, c in columns.items():
            if hasattr(c[0], "__call__"):
                ufunc, c = c
                if ufunc is NewColumn:
                    d = ufunc(c[where])
                else:
                    d = ufunc(self.data[c][where])
            else:
                d = self.data[c][where]
            data[asc] = d
        return NumTable(data) 

    def join(self, other, on, other_on=None, columns=None, other_columns=None, 
            mode=JoinMode.Inner,
            notfound_column="@NA"):
        """ Left join of two tables by 'on'. 
        
            Parameters
            ----------
            other : NumTable
                The right side of the Join operator. 
            on    : 
                Column(s) to join
            other_on :
                The corresponding columns in 'other' table, if different from on.
            columns :
                Columns to select from self. To rename, use a dict of {newname, oldname}.
            other_columns :
                Columns to select from other. 
            notfound_column : string
                Column to store the not found flag. (True for items that do not exist
                in other).

            mode : JoinMode.Inner, JoinMode.Left
                Inner preserves items that exist in both tables.
                Left preserves items that exist in self. notfound_column is set to True
                    if for items does not exits in other.

            Returns
            -------
            NumTable

            Notes
            -----
            Outer join is not supported.
            
            Right join is not supported. Use other.join(self).
             
            For Left join, the columns on non-existing items will have wrong but valid-looking
            values, since numpy does not have NULL. Use notfound_column to detect anormalies.

        """
        on = ensure_tuple(on)

        if other_on is None:
            other_on = on
        if columns is None: columns = tuple(iter(self.data))
        if other_columns is None: other_columns = tuple(iter(other.data))

        other_on = ensure_tuple(other_on)
        columns = ensure_dict(columns)
        other_columns = ensure_dict(other_columns)

        if mode == JoinMode.Right:
            # use a Left join with reserved order,
            # but this would change the renaming scheme.
            raise NotImplemented
        if mode == JoinMode.Outer:
            raise NotImplemented

        if other_on not in other.indices:
            i2 = Index(other.data, other_on, Indexed)
        else:
            i2 = other.indices[other_on]

        # find the matching argindex in other
        arg, notfound = i2.findeach(self.data.toarray(on))

        # prepare the output 
        data = []

        if mode == JoinMode.Inner:
            where = ~notfound
        else:
            # store the notfound field
            # because NULL is not a valid type in numpy
            # these rows will have some valid looking values.
            data.append((notfound_column, notfound ))
            where = Ellipsis

        for asc, c in columns.items():
            data.append((asc, self.data[c][where]))

        for asc, c in other_columns.items():
            # FIXME: skip other_on
            if other_on == on and c in other_on: continue

            data.append((asc, (other.data[c][arg])[where]))

        return NumTable(data)

    def groupby(self, on, aggregations):
        """ groupby operation

            Parameters
            ----------
            on : 
                column(s) to group by

            aggregations: dict
                Mapping in form of {column : (agg, data)}

                Call aggregation for each group on data, the result is 
                stored as column. The aggregation is a callable with
                a signature of :py:code:`agg(array, offsets, axis)` that is
                similar to :py:code:`numpy.ufunc.reduceat`. 
                
                Predefined Aggregations are listed in numdb.Aggregations.
            
            Examples
            --------

            >>> table.groupby(
                    on='Department', 
                    aggregations={
                        'TotalSales' : (numdb.sum, 'Sales'),
                        'MaxSales' : (numdb.max, 'Sales'),
                    })

            Returns
            -------
            NumTable

        """
        on = ensure_tuple(on)
        
        if on not in self.indices:
            i2 = Index(self.data, on, Indexed)
        else:
            i2 = self.indices[on]

        agg = []
        columns = []
        for c, a in aggregations.items():
            agg.append(a)
            columns.append(c)

        offset, result = i2.groupby(agg)

        # prepare the output 
        data = []
        for c in on:
            data.append((c, self.data[c][offset]))

        for c, r in zip(columns, result): 
            data.append((c, r))

        return NumTable(data)

    def __repr__(self):
        s = ("<NumTable Object at 0x%X>\n" % id(self)
            + "- %d Rows" % self.size
            +   " %d Columns :\n" % len(self.columns())
            +
                    '\n'.join(['  %s : %s : %s ...' % 
                        (key, str(self.data.dtype(key)), str(self.data[key][:8]).strip()[:-1]) for key in self.data])
            )
        if len(self.indices):
            s += "\n- Indices:\n"
            s +=  '\n'.join(['  %s : %s' % 
                        (str(key), str(ind.spec)) for key, ind in self.indices.items()])

        return s

class Index(object):
    """ Indexing ndarray.

    """
    def __init__(self, data, columns, spec):
        self.columns = columns
        self.spec = spec
        for column in self.columns:
            if len(data.dtype(column).shape) > 0:
                warnings.warn("Non-scalar columns are indexed as bytes.")

        if data.size < numpy.iinfo(numpy.uint32).max:
            itype = numpy.uint32
        else:
            itype = numpy.uint64

        data_arr = data.toarray(self.columns)

        self.dtype = data_arr.dtype

        arg = data_arr.argsort(order=columns)
        self.indices = numpy.arange(len(data_arr), dtype=itype)

        self.indices = self.indices[arg]
        self.arg = arg
        data_arr = data_arr[arg]
        self.sorteddata = data_arr
        if spec & Unique:
            if len(self.sorteddata) > 1:
                if (self.sorteddata[1:] == self.sorteddata[:-1]).any():
                    raise ValueError("Unique column is not unique")

    def _create_foo(self, value):
        if not isinstance(value, numpy.ndarray):
            foo = numpy.empty((), self.dtype)
        else:
            foo = numpy.empty(len(value), self.dtype)

        for column in self.columns:
            if len(self.columns) > 1:
                foo[column] = value[column]
            else:
                foo[column] = value
        return foo 

    def find(self, value):
        """ find a single item.
            
            Returns
            -------
            integer :
                Index of the item

            Raise
            -----
            ValueError :
                value is not found in the index

        """

        foo = self._create_foo(value)
        arg = self.sorteddata.searchsorted(foo)
        arg = numpy.clip(arg, 0, len(self.sorteddata) - 1)
        notfound = self.sorteddata[arg] != foo 
        if notfound:
            raise ValueError("Value not found")
        return self.indices[arg]

    def findall(self, value):
        foo = self._create_foo(value)
        return self.indices[
                self.sorteddata.searchsorted(foo, side='left'):
                self.sorteddata.searchsorted(foo, side='right')
                ]


    def findeach(self, value):
        """ find many items.

            Returns
            -------
            indices: ndarray
                index of the items. undefined value if not found.
            notfound: ndarray
                True if value is not found.
        """
        foo = self._create_foo(value)
        arg = self.sorteddata.searchsorted(foo)
        arg = numpy.clip(arg, 0, len(self.sorteddata) - 1)
        notfound = self.sorteddata[arg] != foo 
        return self.indices[arg], notfound

    def select(self, left, right):
        left = self._create_foo(left)
        right = self._create_foo(right)
        return self.indices[
                self.sorteddata.searchsorted(left, side='left'):
                self.sorteddata.searchsorted(right, side='right')
                ]

    def groupby(self, aggregations):
        mask = numpy.empty(len(self.sorteddata), '?')
        mask[0] = True
        mask[1:] = self.sorteddata[1:] != self.sorteddata[:-1]
        offsets = numpy.nonzero(mask)[0]

        result = []
        for agg, a in aggregations:
            r = agg(a[self.arg], offsets, axis=0)
            result.append(r)
        return self.indices[offsets], result

## ##
#   Internal functions

def ensure_tuple(columns):
    if not isinstance(columns, (tuple, list, set)):
        columns = (columns, )
    else:
        columns = tuple(columns)
    return columns        

def ensure_dict(columns):
    if isinstance(columns, dict):
        return columns
    columns = ensure_tuple(columns)
    columns = dict(
        [(c, c) for c in columns])
    return columns        

def resolve_name_conflict(data, name):
    c_ = name
    i = 1
    while c_ in data:
        c_ = name + ("_%d" % i)
        i = i + 1
    return c_

