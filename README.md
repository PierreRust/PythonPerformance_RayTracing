

Ray Tracer
==========

A simple ray tracer in python.

Developed to demonstrate profiling tools and performance improvements and presented at [Breizhcamp 2019](https://www.breizhcamp.org/)



Branches
--------

Each branch demonstrates one possible optimization

* **master:** initial basic implementation
* **parallel:** parallel execution with thread or processes
* **intersection:** sublinear intersection test, usinng a KD-tree
* **vector:** replacing the ``Vector3`` class with plain tuples
* **cython:** using cython to move hot code to native compilation
* **numpy_parallel:** vectorization of computation