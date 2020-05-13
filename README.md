# FrozenPy (under construction)

FrozenPy is a small suit of Python functions for detecting freezing behavior and averaging data based on paradigm structure, with a particular focus on Pavlovian conditioning paradigms. Freezing is detected by thresholding motion data under a defined value (e.g. 10 a.u.) for a defined minimum length of time (1 sec). It also includes functions for converting .out files generated from MedPC to easier-to-handle .csv files.

FrozenPy is designed so that it is easy to add metadata (group, sex, etc.) and formats data for use with popular plotting (Seaborn) and statistical (Pingouin) packages within Python.

## Usage

TODO: Create an example which plots raw movement and highlights detected freezing

## Notes

This code was developed specifically for the [Maren Lab](http://marenlab.org/ "Maren Lab homepage") which uses MedPC boxes that measure motion via loadcells, but it should work with any motion data so long as it is in the correct format. If you notice any problems or wish to contribute please don't hesitate to contact me at mictott@gmail.com, open a pull request, or submit an issue.

## Future directions

* plot CS data
* take advantage of xarrays
* provide visible feedback to allow for threshold adjustments (not in the near future unless needed)
