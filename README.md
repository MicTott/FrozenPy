# FrozenPy

FrozenPy is a small collection of Python functions for detecting freezing behavior and averaging data based on a threshold and experimental parameters, with a particular focus on Pavlovian conditioning paradigms. Freezing is detected by thresholding motion data under a defined value (e.g., 10 a.u.) for a defined minimum length of time (1 sec). It also includes functions for converting .out files generated from MedPC to easier-to-handle .csv files.

FrozenPy is designed so that it is easy to add metadata (group, sex, etc.) and formats data for use with popular plotting (Seaborn) and statistical (Pingouin) packages within Python.

## Usage

#### Installation

FrozenPy can easily be installed via [pip](https://pip.pypa.io/en/stable/installing/). Type the following into your terminal to install FrozenPy.

```Python
pip install FrozenPy
```

#### Read .out files
Converting .out to .raw.csv, read .raw.csv:
```Python
# Base directory containing .out files
out_dir = '/path/to/your/.out/files'

# convert all .out files within dir to .raw.csv
fp.read_out(out_dir)

# read .raw.csv
data_raw = fp.read_rawcsv('your_data.raw.csv')
```

#### Detect freezing and average
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

This would output two variables: ```frz_bl``` which contained the averaged BL data for each subject, and ```frz_trials``` which contained CS, US, and ISI data for each subject. These are separated because BL is factorial data whereas Trials are repeated measures.

## Notes

This code was developed specifically for the [Maren Lab](http://marenlab.org/ "Maren Lab homepage") which uses MedPC boxes that measure motion via loadcells, but it should work with any motion data so long as it is in the correct format. If you notice any problems or wish to contribute please don't hesitate to contact me at mictott@gmail.com, open a pull request, or submit an issue.

## Future directions

* take advantage of xarrays (not in the near future)
* provide visible feedback to allow for threshold adjustments (not in the near future unless needed)
