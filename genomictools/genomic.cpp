#include <vector>
#include <algorithm>
#include <iostream>
#include <set>
#include <tuple>
#include <unordered_set>
#include <map>
#include <cstdlib>
#include <Python.h>
#include "genomic.h"
#include <chrono>
using namespace std;

namespace genomic {
	struct base_range_start_compare {
		bool operator() (const BaseRange& lhs, const BaseRange& rhs) const {
			return lhs.start < rhs.start;
		}
	};
	
	bool myfunction2(BaseRange* i, BaseRange* j) {
		return (i->start < j->start) || ((i->start == j->start) && (i->stop < j->stop));
	}
	bool sort_by_small_start_then_large_stop(BaseRange* i, BaseRange* j) {
		return (i->start < j->start) || ((i->start == j->start) && (i->stop > j->stop));
	}
	void FastRangeLookUp::create(std::vector<GenomicPosHolder*> regions) {
		std::sort(regions.begin(), regions.end(), sort_by_small_start_then_large_stop);
		std::vector<std::vector<const GenomicPosHolder*>> joints;
		int layers[regions.size()];
		int k = 0;
		// for (std::vector<GenomicPosHolder>::const_iterator region = regions.begin(); region != regions.end(); region++) {
		for (auto region: regions) {
			int i = 0;
			while (i < this->joints.size() && (this->joints[i].empty() || region->start <= this->joints[i].back()->stop)) {
				i++;
			}
			if (i == this->joints.size()) {
				this->joints.push_back(std::vector<const GenomicPosHolder*>());
			}
			this->joints[i].push_back(region);
			layers[k] = i;
			k++;
		}
// 		cout << "Done part I\n";
		
		// update overlaps
		int disjoint_indice[this->joints.size()]; 
		for (int x = 0; x < this->joints.size(); x++) // ...initialize it
		{
			disjoint_indice[x] = 0;
		}

		for (int j = 0; j < regions.size(); j++){
			int curlayer = layers[j];
			GenomicPosHolder* pr = regions[j];
			// cout<<"ABC "<<pr->start<<','<<pr->stop << " " <<j<<'\n';
// 			if (j % 100000 == 0) {
// 				cout << "Done part II"<<j<<"\n";
// 			}
			std::set<BaseRange, base_range_start_compare> subprs = {BaseRange(pr->start, pr->stop)};
			for (int layer = curlayer + 1; layer < this->joints.size(); layer++) {
				vector<const GenomicPosHolder*>* tmp_overlaps = new vector<const GenomicPosHolder*>();

				int cur_index = disjoint_indice[layer];
			  //  cout << cur_index <<'\n';
				while (cur_index < this->joints[layer].size() && this->joints[layer][cur_index]->stop < pr->start) {
					cur_index++;
				}

				disjoint_indice[layer] = cur_index;
				while (cur_index < this->joints[layer].size() && this->joints[layer][cur_index]->start <= pr->stop) {
					auto target_pr = this->joints[layer][cur_index];
					auto first_it = subprs.upper_bound(*target_pr);
					if (first_it != subprs.begin()) {
						if (prev(first_it)->stop >= target_pr->start) {
							first_it--;
						}
					}
					auto last_it = first_it; 
					while (last_it != subprs.end() && last_it->start <= target_pr->stop) {
						last_it++;
					}

					vector<BaseRange> tmp;
					for (auto it = first_it; it != last_it; it++) {
						tmp_overlaps->push_back(target_pr);
						// 
						if (it->start < target_pr->start) {
							tmp.push_back(BaseRange(it->start, target_pr->start));
						}
						//
						if (it->stop > target_pr->stop) {
							tmp.push_back(BaseRange(target_pr->stop + 1, it->stop));
						}
					}

					subprs.erase(first_it, last_it);
					subprs.insert(tmp.begin(), tmp.end());
					cur_index++;
				}
				if (!tmp_overlaps->empty()) {
					pr->overlapped[layer] = *tmp_overlaps;;
				}
				if (subprs.empty()) {
					break;
				}
			}
		}

	}
	
	bool FastRangeLookUp::overlaps(BaseRange r) {
		auto joints = this->joints;
		vector<const GenomicPosHolder*> j = joints[0];
		auto first_it = upper_bound(j.begin(), j.end(), &r, [](const BaseRange* a, const BaseRange* b){return a->start < b->start;}); 
		if (first_it != j.begin()) {
			first_it--;
		}
		auto it = first_it;
		if (it == j.end()) { // short-circuit true
			return false;
		}
		unordered_set<const GenomicPosHolder*> to_process;
		while (it != j.end() && (*it)->start <= r.stop) {
			if ((*it)->stop >= r.start) { // short-circuit true
				return true;
			}
			to_process.insert(*it);
			it++;
		}
		auto last_it = it;
		
		unordered_set<const GenomicPosHolder*> visited;
		
		
		while (!to_process.empty()) {
			const GenomicPosHolder* pr = *to_process.begin();
			to_process.erase(to_process.begin());
			visited.insert(&*pr);
			if ((pr->start <= r.stop) && (r.start <= pr->stop)) { // short-circuit true
				return true;
			}
			for (auto const& overlap : pr->overlapped) {
				int layer = overlap.first;
				vector<const GenomicPosHolder*> overlaps = overlap.second;

				// auto k = &r;
				auto sub_it = upper_bound(overlaps.begin(), overlaps.end(), &r, 
				 [](const BaseRange* a, const BaseRange* b){return a->start < b->start;}); 
				if (sub_it != overlaps.begin()) {
					sub_it--;
				}
				// cout << "Whatever "<<(*sub_it)<<';'<< (*sub_it)->start << ' ' << (*sub_it)->stop << "\n" ;
				while (sub_it != overlaps.end() && (*sub_it)->start <= r.stop) {
					if (visited.find(*sub_it) == visited.end()) {
						to_process.insert(*sub_it);
					}
					sub_it++;
				}
			}
		}
		return false;
	}
	vector<const GenomicPosHolder*> FastRangeLookUp::query(BaseRange r) {
		auto start1 = std::chrono::high_resolution_clock::now();
		vector<const GenomicPosHolder*>* j = &(this->joints[0]);
		// Find the first it with start smaller 
		auto first_it = upper_bound(j->begin(), j->end(), &r, [](const BaseRange* a, const BaseRange* b){return a->start < b->start;}); 
		if (first_it != j->begin()) {
			first_it--;
		}
		
		auto it = first_it;
		unordered_set<const GenomicPosHolder*> visited;
		unordered_set<const GenomicPosHolder*> to_process;
		unordered_set<const GenomicPosHolder*> results;
		while (it != j->end() && (*it)->start <= r.stop) {
			to_process.insert(*it);
			it++;
		}
		int cter = 0;
		auto last_it = it;
		auto stop1 = std::chrono::high_resolution_clock::now();
		auto start2 = std::chrono::high_resolution_clock::now();
		while (!to_process.empty()) {
			cter++;
			auto start2_1 = std::chrono::high_resolution_clock::now();
			const GenomicPosHolder* pr = *to_process.begin();
			to_process.erase(to_process.begin());
			visited.insert(&*pr);
			if ((pr->start <= r.stop) && (r.start <= pr->stop)) {
				results.insert(&*pr);
			}
			auto stop2_1 = std::chrono::high_resolution_clock::now();
			auto start2_2 = std::chrono::high_resolution_clock::now();
			for (auto const& overlap : pr->overlapped) {
				int layer = overlap.first;
				vector<const GenomicPosHolder*> overlaps = overlap.second;
				// auto k = &r;
				auto sub_it = upper_bound(overlaps.begin(), overlaps.end(), &r, 
				 [](const BaseRange* a, const BaseRange* b){return a->start < b->start;}); 
				if (sub_it != overlaps.begin()) {
					sub_it--;
				}
				while (sub_it != overlaps.end() && (*sub_it)->start <= r.stop) {
					if (visited.find(*sub_it) == visited.end()) {
						to_process.insert(*sub_it);
					}
					sub_it++;
				}
			}
			auto stop2_2 = std::chrono::high_resolution_clock::now();
			this->timing[2] += stop2_1 - start2_1;
			this->timing[3] += stop2_2 - start2_2;

		}
		auto stop2 = std::chrono::high_resolution_clock::now();
		auto start3 = std::chrono::high_resolution_clock::now();
		vector<const GenomicPosHolder*> final_results;
		final_results.insert(final_results.end(), results.begin(), results.end());
		std::sort(final_results.begin(), final_results.end(), [](const BaseRange* a, const BaseRange* b){return a->start < b->start;});
		auto stop3 = std::chrono::high_resolution_clock::now();
		
		this->timing[0] += stop1 - start1;
		this->timing[1] += stop2 - start2;
		this->timing[4] += stop3 - start3;
		
		
		
		return final_results;
	}
	
	void FastRangeLookUp::print_sth() {
		for (auto i = this->joints.begin(); i != this->joints.end(); i++) {
			for (auto a: *i) {
				std::cout<<i - this->joints.begin()<<':' << a->start << ',' << a->stop <<',' << a->overlapped.size()<< ';' << &(*a) <<'\n';
				for (auto const& overlap: a->overlapped) {
					for (auto gp : overlap.second) {
						cout << "    " << overlap.first << ',' <<gp << ','<<gp->start<<','<<gp->stop<<'\n';
					}
				}
			}
			cout << '\n';
		}
	}
	void FastRangeLookUp::print_time() {
	 
		cout << (std::chrono::duration_cast<std::chrono::microseconds>(this->timing[0]).count()) << '\n';
		cout << (std::chrono::duration_cast<std::chrono::microseconds>(this->timing[1]).count()) << '\n';
		cout << (std::chrono::duration_cast<std::chrono::microseconds>(this->timing[2]).count()) << '\n';
		cout << (std::chrono::duration_cast<std::chrono::microseconds>(this->timing[3]).count()) << '\n';
		cout << (std::chrono::duration_cast<std::chrono::microseconds>(this->timing[4]).count()) << '\n';
	}

}
using namespace genomic;
int main() {

	srand (1);
	for (int loop = 0; loop < 1; loop++) {
		vector<GenomicPosHolder*> b;

		for (int i = 0; i < 1; i++) {
			int random_start = rand() % 100 + 1;
			int random_size = rand() % 50 + 10;
			b.push_back(new GenomicPosHolder(random_start, random_start + random_size, NULL));
		}
		
		FastRangeLookUp fdr = FastRangeLookUp();
		fdr.create(b);
		for (int j = 0; j <1000; j++) {
			auto q = BaseRange(80,120);
			auto visitedg = fdr.query(q); //100
		}
		for (auto p : b) {
			delete p;
		}
		b.clear();
// 		fdr.print_time();
		/*
		auto visitedg = fdr.query(q); //100
		unordered_set<const GenomicPosHolder*> true_ans;
		for (auto pr: b) {
		   if ((pr->start <= q.stop) && (q.start <= pr->stop)) {
			   true_ans.insert(pr);
		   }
		}
		*/
// 		cout << (true_ans == visitedg) <<'\n';
	}       
}
