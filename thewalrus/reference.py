# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reference implementations
=========================

.. currentmodule:: thewalrus.reference

This submodule provides access to pure-Python reference implementations of the
hafnian and loop hafnian by summing over the set of perfect matching permutations
or the set of single pair matchings.

For more details on these definitions see:

* Andreas Björklund, Brajesh Gupt, and Nicolás Quesada. "A faster hafnian formula for
  complex matrices and its benchmarking on the Titan supercomputer"
  `arxiv:1805.12498 (2018) <https://arxiv.org/abs/arxiv:1805.12498>`_


Reference functions
-------------------

.. autosummary::
    hafnian

Details
^^^^^^^

.. autofunction::
    hafnian

Auxiliary functions
-------------------

.. autosummary::
    memoized
    partitions
    spm
    pmp
    T

Details
^^^^^^^
"""
import functools

# pylint: disable=too-many-arguments
from collections import OrderedDict
from itertools import tee
from types import GeneratorType

MAXSIZE = 1000
Tee = tee([], 1)[0].__class__


class LimitedSizeDict(OrderedDict):  # pragma: no cover
    r"""Defines a limited sized dictionary.
    Used to limit the cache size.
    """

    def __init__(self, *args, **kwargs):
        self.size_limit = kwargs.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwargs)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


def memoized(f, maxsize=MAXSIZE):
    r"""Decorator used to memoize a generator.

    The standard approach of using ``functools.lru_cache``
    cannot be used, as it only memoizes the generator
    object, not the results of the generator.

    See https://stackoverflow.com/a/10726355/10958457 for the
    original implementation.

    Args:
        f (function or generator): function or generator to
            memoize
        maxsize (int): positive integer that defines
            the maximum size of the cache

    Returns:
        function or generator: the memoized function or generator
    """
    cache = LimitedSizeDict(maxsize=maxsize)

    @functools.wraps(f)
    def ret(*args):
        if args not in cache:
            cache[args] = f(*args)
        if isinstance(cache[args], (GeneratorType, Tee)):
            # the original can't be used any more,
            # so we need to change the cache as well
            cache[args], r = tee(cache[args])
            return r
        return cache[args]

    return ret


@memoized
def partitions(s, singles=True, pairs=True):
    r"""Returns the partitions of a tuple in terms of pairs and singles.

    Args:
        s (tuple): a tuple representing the (multi-)set that will be partitioned.
            Note that it must hold that ``len(s) >= 3``.
        single (boolean): allow singles in the partitions
        pairs (boolean): allow pairs in the partitions

    Returns:
        generator: a generators that goes through all the single-double
        partitions of the tuple
    """
    # pylint: disable=too-many-branches
    if len(s) == 2:
        if singles:
            yield (s[0],), (s[1],)
        if pairs:
            yield s
    else:
        # pull off a single item and partition the rest
        if singles:
            if len(s) > 1:
                item_partition = (s[0],)
                rest = s[1:]
                rest_partitions = partitions(rest, singles, pairs)
                for p in rest_partitions:
                    if isinstance(p[0], tuple):
                        yield ((item_partition),) + p
                    else:
                        yield (item_partition, p)
            else:
                yield s
        # pull off a pair of items and partition the rest
        if pairs:
            # idx0 = 0
            for idx1 in range(1, len(s)):
                item_partition = (s[0], s[idx1])
                rest = s[1:idx1] + s[idx1 + 1 :]
                rest_partitions = partitions(rest, singles, pairs)
                for p in rest_partitions:
                    if isinstance(p[0], tuple):
                        yield ((item_partition),) + p
                    else:
                        yield (item_partition, p)


@memoized
def T(n):
    r"""Returns the :math:`n` th telephone number.

    They satisfy the recursion relation :math:`T(n) = T(n-1)+(n-1)T(n-2)` and
    :math:`T(0)=T(1)=1`.

    See https://oeis.org/A000085 for more details.

    Args:
        n (integer): index

    Returns:
        int: the nth telephone number
     """
    if n in (0, 1):
        return 1
    return T(n - 1) + (n - 1) * T(n - 2)


def spm(s):
    r"""Generator for the set of single pair matchings.

    Args:
        s (tuple): an input tuple

    Returns:
        generator: the set of single pair matching of the tuple s
    """

    def clone_if_single(x):
        """Given an tuple, if its length is one returns a tuple with
        it first and only entry repeated twice. Otherwise it returns the same object.
        """
        if len(x) == 1:
            return (x[0], x[0])
        return x

    for p in partitions(s):
        yield tuple(clone_if_single(i) for i in p)


def pmp(s):
    r"""Generator for the set of perfect matching permutations.

    Args:
        s (tuple): an input tuple

    Returns:
        generator: the set of perfect matching permutations of the tuple s
    """
    return partitions(s, False, True)


def hafnian(M, loop=False):
    r"""Returns the (loop) hafnian of the matrix :math:`M`.

    :math:`M` can be any two-dimensional object of square shape :math:`m` for
    which the elements ``(i, j)`` can be accessed via Python indexing ``M[i, j]``,
    and for which said elements have well defined multiplication ``__mul__``
    and addition `__add__` special methods.

    For example, this includes nested lists and NumPy arrays.

    In particular, one can use this function to calculate symbolic hafnians
    implemented as SymPy matrices.

    Args:
        M (array): a square matrix
        loop (boolean): if set to ``True``, the loop hafnian is returned

    Return:
        scalar: The (loop) hafnian of M
    """
    n, m = M.shape

    if n != m:
        raise ValueError("Input matrix must be square.")

    if n == 0:
        return 1

    if n == 1:
        if loop:
            return M[0, 0]

        return 0

    if n == 2:
        if loop:
            return M[0, 1] + M[0, 0] * M[1, 1]

        return M[0, 1]

    if loop:
        iter_set = spm(tuple(range(n)))
    else:
        iter_set = pmp(tuple(range(n)))

    tot_sum = 0

    for i in iter_set:
        result = 1

        for j in i:
            result = result * M[j]

        tot_sum = tot_sum + result

    return tot_sum
