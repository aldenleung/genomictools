import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

if "--usepyx" in sys.argv:
	extensions = Extension('genomictools', 
				sources = ['genomictools/__init__.pyx'],
				language="c++",
				extra_compile_args=["-std=c++11"],
				extra_link_args=["-std=c++11"],
				)
	sys.argv.remove("--usepyx")
else:
	extensions = Extension('genomictools', 
				sources = ['genomictools/__init__.cpp'],
				language="c++",
				extra_compile_args=["-std=c++11"],
				extra_link_args=["-std=c++11"],
				)
				

with open("README.md", "r") as readme_file:
	readme = readme_file.read()

requirements=[]
setup(
    name='genomictools', 
    version="0.0.6",
    author="Alden Leung",
    author_email="alden.leung@gmail.com",
    description="Tools for processing genomic ranges",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
	ext_modules = cythonize(extensions, compiler_directives={'embedsignature': True}, language_level= 3)
)
