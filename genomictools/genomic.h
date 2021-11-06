#ifndef HAHAHA
#define HAHAHA

#include <vector>
#include <algorithm>
#include <iostream>
#include <set>
#include <tuple>
#include <unordered_set>
#include <map>
#include <cstdlib>
#include <Python.h>
#include <chrono>
namespace genomic {
	class BaseRange {
		public:
			int start;
			int stop;
			BaseRange(int start, int stop) : start(start), stop(stop){};
	};

	class GenomicPosHolder: public BaseRange {
		//using BaseRange::BaseRange;
		public:
			std::map<int, std::vector<const GenomicPosHolder*>> overlapped;
			PyObject* pyObject;
			GenomicPosHolder(int start, int stop, PyObject* pyObject) : BaseRange(start, stop), pyObject(pyObject){};
	};
	class FastRangeLookUp {
		private:
			std::chrono::duration<double> timing[5];
			std::vector<std::vector<const GenomicPosHolder*>> joints;
			
		public:
			void create(std::vector<GenomicPosHolder*> regions);
			bool overlaps(BaseRange r);
			std::vector<const GenomicPosHolder*> query(BaseRange r);

			void print_sth();
			void print_time();
	};
}
#endif