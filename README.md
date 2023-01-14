# genomictools - Tools for processing genomic ranges

The genomictools package provides an easy solution to all data types with a genomic range.  



## Installation 

```
pip install genomictools
```



## Basic usage

The basic class of a genomic range is called `GenomicPos`. The start and stop in GenomicPos are all 1-based coordinate. 

```python
r = GenomicPos("chr1", 1, 100)
print(str(r), r.name, r.start, r.stop) # "chr1:1-100" "chr1", 1, 100
r = GenomicPos("chr1:1-100")
print(str(r), r.name, r.start, r.stop) # "chr1:1-100" "chr1", 1, 100
r = GenomicPos("chr1:1")
print(str(r), r.name, r.start, r.stop) # "chr1:1-1" "chr1", 1, 1

```

To avoid confusion, it also provides the way to get 0-based or 1-based coordinate `zstart`, `ostart`, `zstop`, `ostop`. For example, in BED, we usually have 0-based start coordinate and 1-based stop coordinate. 

```python
r = GenomicPos("chr1", 1, 100)
print(r.name, r.zstart, r.ostop) # "chr1" 0 100
```



To store a list of genomic ranges, we can use `GenomicCollection`.

```python
from genomictools import GenomicPos, GenomicCollection

r1 = GenomicPos("chr1:1-100")
r2 = GenomicPos("chr3:1000-2000")
r3 = GenomicPos("chr1:51-200")

regions = GenomicCollection([r1, r2, r3])
print(len(regions)) # 3

# When iterating through the regions, they will be sorted by name, start and stop. 
for r in regions: 
	print(str(r))
# chr1:1-100
# chr1:51-200
# chr3:1000-2000

# One can check if a region overlaps any entry within the genomic collection
print(regions.overlaps(GenomicPos("chr1:201-300"))) # False
print(regions.overlaps(GenomicPos("chr1:2-3"))) # True
print(regions.overlaps(GenomicPos("chr2:2-3"))) # False

# One can extract all entries from the genomic collection that overlap with the target region
for r in regions.find_overlaps(GenomicPos("chr1:1-3")):
	print(str(r))
# chr1:1-100
for r in regions.find_overlaps(GenomicPos("chr1:26-75")):
	print(str(r))
# chr1:1-100
# chr1:51-200


```



For any data entry with an associated genomic range, it will implement `GenomicAnnotation`, where any `GenomicAnnotation` instance will have a property `genomic_pos`

```python
from biodata.bed import BED
bed = BED("chr1", 0, 100, name="R1")
print(str(bed.genomic_pos)) # chr1:1-100
```



One can use  `GenomicAnnotation` as entries in `GenomicCollection`. 

```python
from biodata.bed import BED
from genomictools import GenomicPos, GenomicCollection

beds = GenomicCollection([BED("chr1", 0, 100, name="R1"), BED("chr3", 1999, 2000, name="R2"), BED("chr1", 50, 200, name="R3")])
for bed in beds:
	r = bed.genomic_pos
	print(bed.name, str(r))
# R1 chr1:1-100
# R3 chr1:51-200
# R2 chr3:2000-2000

for bed in beds.find_overlaps(GenomicPos("chr1:26-75")):
	r = bed.genomic_pos
	print(bed.name, str(r))
# R1 chr1:1-100
# R3 chr1:51-200

```



The base class of genomic range with strand is `StrandedGenomicPos`, which extends `GenomicPos`. 

For any data entry with an associated stranded genomic range, it will implement `StrandedGenomicAnnotation`, where any `StrandedGenomicAnnotation` instance will have properties `stranded_genomic_pos` and `genomic_pos`. The strand should be `+` for positive strand, `-` for negative strand and `.` for unspecified strand. 

```python
from biodata.bed import BED
bed = BED("chr1", 0, 100, name="R1", strand="-")
print(str(bed.stranded_genomic_pos)) # chr1:1-100:-
print(str(bed.genomic_pos)) # chr1:1-100
```



One could also use `GenomicCollection` to store genomic ranges data easily:

```python
from biodata.bed import BEDReader
from genomictools import GenomicCollection
beds = BEDReader.read_all(GenomicCollection, filename)
```

