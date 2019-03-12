TODO
====

ISSUES
------

* bounding box for plane ???

FEATURES
--------
* [X] Reorganize sources
* [X] Command line interface
* [X] Yaml format for scene
* [ ] Gui: display image during computation (gtk + canvas)
* [ ] Replace Phong model with Blinn Phong : faster
* [ ] Use Beer's Law for light attenuation inside material
* [ ] Scale of light power ? apply automatic normalization ?
* [ ] Other types of light sources : directional sources, area sources,
     environment sources
* [ ] Shadow projected by transparent object ?
* [ ] Anti-aliasing
* [ ] Soft shadows
* [ ] Depth of field
* [ ] Texture : image on surface
* [ ] Other shapes : triangle, rectangle, tor
* [X] Transparency - aka refraction
* [X] Add mirror reflection
* [X] Light intensity and color => through light power & distance
* [X] Take into account light source distance


IMPROVE / REFACTORING
---------------------

* [ ] Replace vector by tuple + functions : easier drop-in replacement for numpy
* [ ] No need to return several points for intersection  ?
* [ ] Create a structure for surface properties ? with some predefined constants
* [ ] Only return t, point could be computed when needed ?

PERFORMANCE
-----------

* [X] Bounding box algorithm for intersection : KD tree
* [ ] Mask to avoid computing all pixels
* [ ] Replace Vector class with ndarray
* [ ] Matrix computation with numpy
* [ ] Test with pypy
* [ ] Test with pypy
