from cpython.ref cimport PyObject
from libcpp.string cimport string
from libcpp.map cimport map as mapcpp

import cython
import heapq
import re
from collections import defaultdict

from genomictools.genomic cimport BaseRange
from genomictools.genomic cimport GenomicPosHolder, FastRangeLookUp, vector

__all__ = ["GenomicAnnotation", "GenomicPos", "AbstractGenomicCollection", "GenomicCollection", "union", "intersection", "substract"]


_genomic_pos_pattern = re.compile("([^:]+):(-?[0-9]+)(?:-(-?[0-9]+))?") # Used in parsing GenomicPos string
@cython.auto_pickle(False)
cdef class GenomicAnnotation():
	'''
	Any object that contains a representation of the genomic position.
	'''
	@property
	def genomic_pos(self):
		'''
		The GenomicPos representation of this object
		'''
		pass

@cython.binding(True)
@cython.auto_pickle(True)
cdef class GenomicPos(GenomicAnnotation):
	'''
	A class that represents a genomic position. The start and stop are all 1-based coordinate. 
	
	:Example:

	.. code-block:: python
	
		g1 = GenomicPos("chr1", 1, 1000)
		g2 = GenomicPos("chr1", 501, 1500)
		g3 = GenomicPos("chr1", 2001, 3000)
		
	'''

	cdef str name
	cdef int zstart
	cdef int ostop

	def __init__(self, name, start=None, stop=None):
		if start is None:
			if hasattr(name, "genomic_pos"):
				r = name.genomic_pos
				name = r.name
				start = r.start
				stop = r.stop
			else:
				try:
					name, start, stop = _genomic_pos_pattern.match(name).groups()
					start = int(start)
					if stop is not None:
						stop = int(stop)
				except:
					raise Exception("Cannot parse genomic_pos str " + name)
		self.name = name
		if stop is None:
			stop = start
		self.zstart = start - 1
		self.ostop = stop
	
	cpdef bint overlaps(self, GenomicPos other):
		'''
		check if this GenomicPos overlaps with the other GenomicPos
		'''
		return self.name == other.name and self.ostop > other.zstart and other.ostop > self.zstart
	
	
	def __hash__(self):
		return hash((self.name, self.start, self.stop))

	def __len__(self):
		return self.ostop - self.zstart
	

	def __lt__(self, other):
		if not isinstance(other, GenomicPos):
			return NotImplemented
		return ((self.name < other.name) 
				or (self.name == other.name and 
					((self.zstart < other.zstart) or (self.zstart == other.zstart and self.ostop < other.ostop))))
	
	def __eq__(self, other):
		if not isinstance(other, GenomicPos):
			return NotImplemented
		return self.zstart == other.zstart and self.ostop == other.ostop and self.name == other.name
	
	def __contains__(self, GenomicPos target_pos):
		return self.name == target_pos.name and self.zstart <= target_pos.zstart and self.ostop >= target_pos.ostop
	
	def __str__(self):
		return self.name + ":" + str(self.ostart) + "-" + str(self.ostop)
		
	@property
	def name(self):
		'''
		The chromosome name
		'''
		return self.name
	
	@property
	def zstart(self):
		'''
		The start position (0-based)
		'''
		return self.zstart
	
	@property
	def zstop(self):
		'''
		The stop position (0-based)
		'''
		return self.ostop - 1
	
	@property
	def ostart(self):
		'''
		The start position (1-based)
		'''
		return self.zstart + 1
	
	@property
	def ostop(self):
		'''
		The stop position (1-based)
		'''
		return self.ostop
	
	@property
	def start(self):
		'''
		The start position (1-based)
		'''
		return self.ostart

	@property
	def stop(self):
		'''
		The stop position (1-based)
		'''
		return self.ostop
	
	@property
	def genomic_pos(self):
		'''
		The GenomicPos representation of this object
		'''
		return self


cdef class AbstractGenomicCollection():
	'''
	A base class for all GenomicCollection implementations.
	'''
# 	@abc.abstractmethod
	def __iter__(self):
		'''
		An iterator of sorted regions
		'''
		raise NotImplementedError
	
# 	@abc.abstractmethod
	def __len__(self):
		'''
		Number of entries the collection holds
		'''
		raise NotImplementedError
	
	def overlaps(self, q):
		'''
		Deterimine whether this collection overlaps with any regions indicated in `q`. 
		'''
		return any(True for _ in self.find_overlaps(q))
	
	def find_overlaps(self, q):
		'''
		Find all GenomicAnnotation in this collection that overlaps with `q`
		
		Return a generator of all the overlapped regions
		'''
		for r in iter(self):
			if r.genomic_pos.overlaps(q.genomic_pos):
				yield r
	
	def find_non_overlaps(self, q):
		'''
		Find all GenomicAnnotation in this collection that does not overlap with q
		
		Return a generator of all the non-overlapping regions
		'''
		for r in iter(self):
			if not r.genomic_pos.overlaps(q.genomic_pos):
				yield r
	
	def add(self, r):
		'''
		Add a region or regions to this collection. If the collection architecture does not support addition, an NotImplementedError is raised.
		'''
		raise NotImplementedError
		
	def remove(self, r):
		'''
		Remove a region or regions from this collection. If the collection architecture does not support removal, an NotImplementedError is raised.
		'''
		raise NotImplementedError

# cdef class ListGenomicCollection(AbstractGenomicCollection):
	# '''
	# A basic genomic collection that stores all the annotated regions
	# '''
	# cdef list regions
	# def __init__(self, iterable):
		# self.regions = sorted(list(iterable))
	# def __iter__(self):
		# return iter(self.regions)
	# def __len__(self):
		# return len(self.regions)

		

@cython.binding(True)
cdef class GenomicCollection(AbstractGenomicCollection):
	cdef mapcpp[string, FastRangeLookUp] frs
	cdef list regions
	def __init__(self, regions):
		self.regions = sorted(regions, key=lambda r: r.genomic_pos)
		sep_rs = defaultdict(list)
		for r in self.regions:
			sep_rs[r.genomic_pos.name].append(r)
		cdef vector[GenomicPosHolder*] holders
		for chrname, rs in sep_rs.items():
			for r in rs:
				holders.push_back(new GenomicPosHolder(r.genomic_pos.start, r.genomic_pos.stop, <PyObject*>r))
			
			self.frs[chrname.encode('utf-8')] = FastRangeLookUp()
			self.frs[chrname.encode('utf-8')].create(holders)
			holders.clear()
	
	def __iter__(self):
		return iter(self.regions)
	
	def __len__(self):
		return len(self.regions)

	def _query(self, q):
		if self.frs.find(q.genomic_pos.name.encode('utf-8')) == self.frs.end():
			return []
		holders = self.frs[q.genomic_pos.name.encode('utf-8')].query(BaseRange(q.genomic_pos.start, q.genomic_pos.stop))
		return [<object>i.pyObject for i in holders]
	
	def overlaps(self, q):
		if self.frs.find(q.genomic_pos.name.encode('utf-8')) == self.frs.end():
			return False
		return self.frs[q.genomic_pos.name.encode('utf-8')].overlaps(BaseRange(q.genomic_pos.start, q.genomic_pos.stop))
	
	
	def find_overlaps(self, q):
		yield from self._query(q)
	
	def __reduce__(self):
		return (GenomicCollection, (self.regions,))


@cython.binding(True)
def intersection(*genomic_collections, mergefunc=None):
	'''
	Return the intersection of all genomic collections.
	
	:param genomic_collections: GenomicCollection or an iterator of sorted GenomicPos
	:param mergefunc: Reserved keywords.

	:Example:
	
	.. code-block:: python
	
		from genomictools import GenomicCollection, GenomicPos, intersection
		
		for r in intersection(GenomicCollection([GenomicPos("chr1", 1, 10), GenomicPos("chr1", 15, 20)])):
			print(r.name, r.start, r.stop)
		# chr1 1 10
		# chr1 15 20
		
		for r in intersection(GenomicCollection([GenomicPos("chr1", 1, 10), GenomicPos("chr1", 15, 20)]),
		                      GenomicCollection([GenomicPos("chr1", 8, 12), GenomicPos("chr1", 14, 21)])):
			print(r.name, r.start, r.stop)
		# chr1 8 10
		# chr1 15 20
		
	'''
	def _internal_intersection(refs, queries):
		for q in queries:
			yield from union(GenomicPos(q.genomic_pos.name, max(q.genomic_pos.start, hit.genomic_pos.start), min(q.genomic_pos.stop, hit.genomic_pos.stop)) for hit in refs.find_overlaps(q))
	if len(genomic_collections) == 0:
		return
	
	genomic_collections = sorted(genomic_collections, key=lambda k: len(k))
	giter = iter(genomic_collections)
	intersection_generator = union(next(giter))
	for genomic_collection in giter:
		intersection_generator = _internal_intersection(genomic_collection, intersection_generator)
	yield from intersection_generator

@cython.binding(True)
def union(*genomic_collections, mergefunc=None):
	'''
	Return the union of all genomic collections. 
	
	:param genomic_collections: GenomicCollection or an iterator of sorted GenomicPos
	:param mergefunc: Reserved keywords.
	
	:Example:
	
	.. code-block:: python
	
		from genomictools import GenomicCollection, GenomicPos, union
		
		for r in union(GenomicCollection([GenomicPos("chr1", 1, 10), GenomicPos("chr1", 15, 20), GenomicPos("chr1", 22, 24)]),
		               GenomicCollection([GenomicPos("chr1", 8, 12), GenomicPos("chr1", 14, 21)])):
			print(r.name, r.start, r.stop)
		# chr1 1 12
		# chr1 14 24
	
	'''
	giter = heapq.merge(*genomic_collections, key=lambda k: k.genomic_pos)
	try:
		stored = next(giter)
	except StopIteration:
		# Only if no elements are in these collections
		return
	for r in giter:
		if stored.genomic_pos.name == r.genomic_pos.name and r.genomic_pos.start - stored.genomic_pos.stop <= 1:
			stored = GenomicPos(stored.genomic_pos.name, stored.genomic_pos.start, max(stored.genomic_pos.stop, r.genomic_pos.stop))
		else:
			yield stored
			stored = r
	yield stored
	
@cython.binding(True)
def substract(query, *refs):
	'''
	Return all regions that are present in query but not in refs. 
	
	:param query: GenomicCollection or an iterator of sorted GenomicPos
	:param refs: GenomicCollection
	
	:Example:
	
	.. code-block:: python
	
		from genomictools import GenomicCollection, GenomicPos, substract
		
		for r in substract(GenomicCollection([GenomicPos("chr1", 1, 10), GenomicPos("chr1", 15, 20), GenomicPos("chr1", 22, 24)]),
		               GenomicCollection([GenomicPos("chr1", 8, 12), GenomicPos("chr1", 14, 21)])):
			print(r.name, r.start, r.stop)
		# chr1 1 7
		# chr1 22 24
	
	'''
	ref = GenomicCollection(union(*refs))
	for q in query:
		start_pos = q.genomic_pos.start
		for r in ref.find_overlaps(q):
			if start_pos <= r.genomic_pos.start - 1:
				yield(GenomicPos(q.genomic_pos.name, start_pos, r.genomic_pos.start - 1))
			start_pos = r.genomic_pos.stop + 1
		if start_pos <= q.genomic_pos.stop:
			yield(GenomicPos(q.genomic_pos.name, start_pos, q.genomic_pos.stop))


