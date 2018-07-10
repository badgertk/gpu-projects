# QTc Clock Plotter #

This repository contains a python class that provides simple plotting of ECG data (specifically, QTc values) in the "24 hour clock" format:

![Example QT Clock](https://bitbucket.org/atpage/ecgclock/raw/master/example_data/baseline_vs_drug.png)

### DISCLAIMER: ###

All of the included default values, ranges, example data sets, etc. are for illustration, not for diagnostic use.  It is the clinician's responsibility to adjust all settings appropriately.

### How do I get set up? ###

You will need Python with the following modules available:

* [`numpy`](http://www.numpy.org/)
* [`dateutil`](http://labix.org/python-dateutil)
* [`matplotlib`](http://matplotlib.org/)
* [`cycler`](http://matplotlib.org/cycler/)

Depending on the chosen `matplotlib` backend, there may be other dependencies, such as `PySide` or `PyGTK`.

The code has been tested in Python 2.7 and 3.4.

### How do I run it? ###

See `clock_example.py` or `./QTClock.py -h`.  `clock_example.py` demonstrates most of the features, while directly running `QTClock.py` only provides access to some of them.

### Who do I talk to? ###

* Alex Page, alex.page@rochester.edu
* Jean-Phillippe Couderc, URMC