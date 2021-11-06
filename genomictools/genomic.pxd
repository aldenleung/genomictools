from cpython.ref cimport PyObject
from libcpp cimport bool

from libcpp cimport vector
cdef extern from "<vector>" namespace "std":
	cdef cppclass vector[T]:
		cppclass iterator:
			T operator*()
			iterator operator++()
			bint operator==(iterator)
			bint operator!=(iterator)
		vector()
		void push_back(T&)
		T& operator[](int)
		T& at(int)
		iterator begin()
		iterator end()
		void clear()

cdef extern from "genomic.cpp":
	pass
cdef extern from "genomic.h" namespace "genomic":
	cdef cppclass BaseRange:
		BaseRange(int, int)
	cdef cppclass GenomicPosHolder(BaseRange):
		GenomicPosHolder(int, int, PyObject*)
		int start
		int stop
		PyObject* pyObject
	cdef cppclass FastRangeLookUp:
		FastRangeLookUp()
		void create(vector[GenomicPosHolder*] regions)
		bool overlaps(BaseRange r);
		vector[const GenomicPosHolder*] query(BaseRange r)
		void print_sth()

	
