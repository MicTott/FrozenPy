# FrozenPy

FrozenPy is a small suit of Python functions for detecting freezing behavior and averaging data based on paradigm structure, with a particular focus on Pavlovian conditioning paradigms. Freezing is detected by thresholding motion data under a defined value (e.g. 10 a.u.) for a defined minimum length of time (1 sec). It also includes functions for converting .out files generated from MedPC to easier-to-handle .csv files.

FrozenPy is designed so that it is easy to add metadata (group, sex, etc.) and formats data for use with popular plotting (Seaborn) and statistical (Pingouin) packages within Python.

## Usage

##### Installation

FrozenPy cannot currently be installed via pip, so for if you want to use it you can download or clone this repository to your local machine by clicking ``` Clone or download ``` in the upper right corner,

or it can be cloned using git with:
```
git clone https://github.com/MicTott/FrozenPy
```

##### Importing FrozenPy functions

Because FrozenPy cannot be installed by pip yet, you need to be in the directory containing containing 'FrozenPy.py' to import the functions. This is easily done with the 'os' package:
```Python
import os
path = '/your/path/to/FrozenPy'
os.chdir(path)
import FrozenPy as fp
```

##### Read .out files
Converting .out to .raw.csv, read .raw.csv:
```Python
# Base directory containing .out files
out_dir = '/path/to/your/.out/files'

# convert all .out files within dir to .raw.csv
fp.read_out(out_dir)

# read .raw.csv
data_raw = fp.read_rawcsv('your_data.raw.csv')
```

##### Detect freezing and average
Detect freezing:
```Python
# detect freezing
data_freezing = fp.detect_freezing(data_raw)
```
This is an example for if we wanted to slice and average data with a 3 min baseline, 10s CS, 2s US, 58s ISI, and 5 trials:
```Python
# slice data
frz_bl, frz_trials = fp.get_averagedslices(df=data_freezing,
                                         BL=180,
                                         CS=10,
                                         US=2,
                                         Trials=5,
                                         ISI=58,
                                         fs=5,
                                         Behav='Freezing')
```

This would output two variables: ```frz_bl``` which contained the averaged BL data for each subject, and ```frz_trials``` which contained CS, US, and ISI data for each subject. These are seperated because BL is factorial data whereas Trials are repeated measures. Combining these into one dataframe gets weird in long format and it's easiest to keep these separated for plotting and statistics.

## Notes

This code was developed specifically for the [Maren Lab](http://marenlab.org/ "Maren Lab homepage") which uses MedPC boxes that measure motion via loadcells, but it should work with any motion data so long as it is in the correct format. If you notice any problems or wish to contribute please don't hesitate to contact me at mictott@gmail.com, open a pull request, or submit an issue.

## Future directions

* pip integration
* plot CS data
* take advantage of xarrays
* provide visible feedback to allow for threshold adjustments (not in the near future unless needed)
