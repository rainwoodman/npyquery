"""
Numpy Database

"""
    
# here are a few examples

def test():
    dtype = numpy.dtype(
        [('id', 'i8'),
         ('s1', 'f4'),
        ])

    dtype2 = numpy.dtype(
        [('id', 'i8'),
         ('id2', 'i4'),
         ('s3', 'f4'),
        ])

    data = numpy.empty(5, dtype=dtype)
    data2 = numpy.empty(5, dtype=dtype2)

    for column in data.dtype.fields:
        data[column].flat[:] = numpy.arange(data[column].size)

    for column in data2.dtype.fields:
        data2[column].flat[:] = numpy.arange(data2[column].size)
    data['s1'][2] = 1.0
    data2['id2'][2] = 10

    table = NumTable(
                data=data,
                specs={'id': Unique | Indexed,
                's1' :  Indexed,
                },
            )
    table2 = NumTable(
                data=data2,
                specs={
                'id' : Unique | Indexed,
                's3' : Indexed,
                }
                )

    print 'table', table
    print 'table2', table2
    print 'findall', table2.indices['s3'].findall(2)
    print 'find', table2.indices['s3'].find(2)
    print 'select', table2.select(['id', 's3'], where=table2['id'] != 3)

    j = table.join(table2, on='id', on_other='id2')
    print 'join', j

    j = table.join(table2, on='id', on_other='id2', mode=JoinMode.Left)
    print 'left join', j

    print 'groupby', table.groupby('s1', 
            {'s1' : (numpy.add, table['s1'])})

import numpy
import warnings

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
Unique = IndexSpec.Unique = IndexSpec(4)

class JoinMode(int): 
    pass
JoinMode.Inner = JoinMode(0)
JoinMode.Left = JoinMode(1)
JoinMode.Right = JoinMode(2)
JoinMode.Outer = JoinMode(3)

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
        elif isinstance(data, numpy.ndarray):
            d = dict([
                (key, data[key]) for key in data.dtype.fields])
            return NumData(d)            
        elif isinstance(data, NumData):
            return NumData(data.base)
        elif isinstance(data, (tuple, list, set)):
            self = NumData.concatenate([
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
        return iter(self.base)

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
            data = numpy.empty(len(self), dtype)
            for column in columns:
                data[column][...] = self[column]
        return data
        
    def __getitem__(self, column):
        if not column in self.base:
            raise KeyError('Column `%s` not found' % str(column))
        return self.base[column]
        

    @classmethod
    def concatenate(kls, list):
        data = dict(
            [(key, item[key]) for item in list for key in item])
        return NumData(data) 

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
    while c_ in data:
        c_ = c_ + "_"
    return c_

class NumTable(object):
    """ A relational data table based on numpy array.


        Parameters
        ----------
        array:  ndarray
            
    """

    def __init__(self, data, specs=None):
        self.indices = {}
        self.specs = {}
        self.data = NumData(data)
        if specs is not None:
            self.update_indices(specs)

    def __getitem__(self, columns):
        return self.data.toarray(columns)

    def append_columns(self, data):
        """ Append columns from data.
            
            I hope this is not used very often.
        """
        self.data = NumData.concatenate((self.data, NumData(data)))
 
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
                Columns to select. To rename, use a dict, where 
                value is the new name.
                    
            Returns
            -------
            NumTable

        """
        if columns is None:
            columns = tuple(iter(self.data))

        columns = ensure_dict(columns)        
        if where is None:
            where = Ellipsis
        data = dict([(asc, self.data[c][where]) for c, asc in columns.items()])
        return NumTable(data) 

    def join(self, other, on, on_other=None, columns=None, other_columns=None, 
            mode=JoinMode.Inner,
            notfound_column="@NA"):
        """ Left join of two tables by 'on'. 
        
            Parameters
            ----------
            other : NumTable
                The right side of the Join operator. 
            on    : 
                Column(s) to join
            on_other :
                The corresponding columns in 'other' table, if different from on.
            columns :
                Columns to select from self. To rename, use a dict, where the value is
                the new column name.
            others_column :
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

        if on_other is None:
            on_other = on
        if columns is None: columns = tuple(iter(self.data))
        if other_columns is None: other_columns = tuple(iter(other.data))

        on_other = ensure_tuple(on_other)
        columns = ensure_dict(columns)
        other_columns = ensure_dict(other_columns)

        if mode == JoinMode.Right:
            # use a Left join with reserved order,
            # but this would change the renaming scheme.
            raise NotImplemented
        if mode == JoinMode.Outer:
            raise NotImplemented

        if on_other not in other.indices:
            i2 = Index(other.data, on_other, Indexed)
        else:
            i2 = other.indices[on_other]

        # find the matching argindex in other
        arg, notfound = i2.findeach(self.data.toarray(on))

        # prepare the output 
        data = {}

        if mode == JoinMode.Inner:
            where = ~notfound
        else:
            # store the notfound field
            # because NULL is not a valid type in numpy
            # these rows will have some valid looking values.
            data[notfound_column] = notfound 
            where = Ellipsis

        for c, asc in columns.items():
            data[asc] = self.data[c][where]

        for c, asc in other_columns.items():
            # FIXME: skip on_other
            # if c in on_other: continue
            #
            # append prime for conflicting names
            c_ = resolve_name_conflict(data, asc)
            data[c_] = (other.data[c][arg])[where]

        return NumTable(data)

    def groupby(self, on, aggregations):
        """ groupby operation

            Parameters
            ----------
            on : 
                column(s) to group by
            aggregations: dict
                Mapping in form of {column : (ufunc, data)}

                Reduce ufunc for each group on data, the result is 
                stored as column.
            
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
        data = {}
        for c in on:
            data[c] = self.data[c][offset]

        for c, r in zip(columns, result): 
            c = resolve_name_conflict(data, c)
            data[c] = r

        return NumTable(data)

    def __repr__(self):
        s = ("<NumTable Object at 0x%X>" % id(self)
            + " %d Rows" % len(self.data)
            +   "\n%d Columns :\n" % len(self.data)
            +
                    '\n'.join(['  %s : %s : %s ...' % 
                        (key, str(self.data.dtype(key)), str(self.data[key][:8]).strip()[:-1]) for key in self.data])
            )
        if len(self.indices):
            s += "\nIndices:\n"
            s +=  '\n'.join(['  %s : %s' % 
                        (str(key), str(ind.spec)) for key, ind in self.indices.items()])
        else:
            s += "\nNo indices\n"
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

        if len(data) < numpy.iinfo(numpy.uint32).max:
            itype = numpy.uint32
        else:
            itype = numpy.uint64

        data_arr = data.toarray(self.columns)

        self.dtype = data_arr.dtype

        arg = data_arr.argsort(order=columns)
        self.indices = numpy.arange(len(data_arr), dtype=itype)

        self.indices = self.indices[arg]
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
        mask[0] = False
        mask[1:] = self.sorteddata[1:] != self.sorteddata[:-1]
        offsets = numpy.nonzero(mask)[0]

        result = [ ufunc.reduceat(a, offsets, axis=0)
                    for ufunc, a in aggregations]
        return offsets, result

if __name__ == '__main__':
    test()
