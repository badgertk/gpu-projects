#!/usr/bin/env python

"""This module provides a class and supporting functions for visualizing
long-term QTc monitoring data.
"""

################################### Imports: ###################################

# Need to do this to prevent warnings/errors when saving a batch of clocks:
import matplotlib
matplotlib.use('Qt4Agg')  # adds improved zoom over default backend
# matplotlib.use('GTKCairo')  # switch to vector graphics
# matplotlib.use('GTKAgg')  # nothing fancy over default.  similar to WXAgg.

import numpy as np
import argparse
import multiprocessing as mp

from ECGClock import *
# TODO: may not need '*' once we get this file cleaned up

################################# Main Class: ##################################

class QTClock(ECGClock):
    def __init__(self, title=None,
                 min_rad=10, max_rad=155,
                 color_cycle=['b', 'm', 'c', 'k', 'y'],
                 parent_figure=None,
                 subplot=1):
        """TODO.
        """
        # Save args to pass on:
        locals_minus_self = locals()
        locals_minus_self.pop('self', None)

        #super(QTClock, self).__init__( **locals_minus_self )  # python 2
        #super().__init__( **locals_minus_self )  # python 3?
        ECGClock.__init__(self, **locals_minus_self )  # python 2 or 3, maybe

        self.ax.format_coord = self.format_coord

    def add_default_ranges(self):
        """Highlight the typical 'healthy' and 'danger' ranges on the plot.

        Keyword arguments:
        subplot -- which subplot to add ranges to (1-indexed)
        """
        self.add_danger_range()
        self.add_healthy_range()

    def add_healthy_range(self, qtc_range=[367, 435]):
        """Highlight healthy QTc region on a plot.  zorder is set to -1 for this, so
        foreground items should use zorder>-1.  The default range was chosen to
        cover roughly 1 standard deviation for men and women combined.

        Keyword arguments:
        qtc_range -- range to highlight, e.g. [350, 450]
        """
        theta = np.linspace(0, 2*np.pi, 100)  # 100 may not always be enough

        self.ax.fill_between(theta, qtc_range[0], qtc_range[1],
                             color='green', alpha=0.2, linewidth=0, zorder=-1,
                             label="healthy")
        # TODO: M vs F ranges
        # TODO: color, alpha as args?  maybe make an add_static_range() that
        # both healthy and danger call

    def add_danger_range(self, min_qtc=500, color='red'):
        """Highlight dangerous QTc region on a plot.  zorder is set to -1 for this, so
        foreground items should use zorder>-1.

        Keyword arguments:
        min_qtc -- QTc is dangerous above this value
        color -- color of highlight
        """
        theta = np.linspace(0, 2*np.pi, 100)  # 100 may not always be enough
        self.ax.fill_between(theta, min_qtc, self.ax.get_ylim()[1],
                             color=color, alpha=0.2, linewidth=0,
                             zorder=-1, label="danger")
        # TODO: alpha = param / static var ?  same for color?
        # TODO: handle axes being zoomed out in interactive window?

    def format_coord(self, th, r):
        """Return a human readable string from a (theta, r) coordinate."""
        return 'time=' + angle_to_time(th) + ', QTc=%1.0fms'%(r)  # 'QT' vs. 'QTc'?

# TODO: kwargs in several places?

############################### Extra Functions: ###############################

# (none... QT-specific helper functions would go here)

################################### main(): ####################################

if __name__=='__main__':
    #mp.freeze_support()  # only needed in Windows

    #### Parse input args: ####

    argparser = argparse.ArgumentParser(
        description="Generate a QT Clock.  Input files will be identified by filename in the legend." +
        "  Note that not all features are accessible via the CLI."
    )
    argparser.add_argument("-i", "--input-file",
                           help="input filename(s), e.g. 'qtcb.csv'",
                           nargs='+')
    argparser.add_argument("-o", "--output-file",
                           help="output filename, e.g. 'out.png'")
    argparser.add_argument("-t", "--title",
                           help="plot title")
    argparser.add_argument("-d", "--default-highlights",
                           help="highlight default 'healthy' and 'danger' ranges (1=yes, default)",
                           type=int, choices=[0,1], default=1)
    argparser.add_argument("-f", "--filter",
                           help="width of filter window, minutes (default=0, disabled).  applies to all recordings.",
                           type=float, default=0)
    argparser.add_argument("-r", "--range-file",
                           help="filename(s) for saved QTc ranges, e.g. 'qtcb_healthy_male.csv'",
                           nargs='+')
    argparser.add_argument("--no-legend",
                           help="don't include legend on plot",
                           action="store_true", default=False)
    argparser.add_argument("-s", "--show",
                           help="open plot in interactive window",
                           action="store_true")
    args = argparser.parse_args()

    # TODO: default values where appropriate
    # TODO: args for static healthy and danger ranges
    # TODO: check feature set again... add more important ones, remove unnecessary ones.

    #### Make the clock: ####

    my_clock = QTClock(args.title)
    if (args.input_file):
        for filename in args.input_file:
            my_clock.add_recording(filename,
                                   label=filename,
                                   filtering=args.filter)  # TODO: make label an arg
    if (args.default_highlights):
        my_clock.add_default_ranges()
    if (args.range_file):
        for filename in args.range_file:
            my_clock.add_saved_range(filename, label=filename)  # TODO: make label an arg
    if (not args.no_legend):
        my_clock.add_legend()
    if (args.output_file):
        my_clock.save(args.output_file)
    if (args.show):
        my_clock.show()

#################################### TODO: #####################################

# ...

################################################################################
