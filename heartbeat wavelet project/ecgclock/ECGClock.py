#!/usr/bin/env python

"""This module provides a class and supporting functions for visualizing
features from long-term ECG monitoring data.
"""

################################### Imports: ###################################

# Need to do this to prevent warnings/errors when saving a batch of clocks:
import matplotlib
matplotlib.use('Qt4Agg')  # adds improved zoom over default backend
# matplotlib.use('GTKCairo')  # switch to vector graphics
# matplotlib.use('GTKAgg')  # nothing fancy over default.  similar to WXAgg.

#import matplotlib.pyplot as plt
import numpy as np
import csv
from dateutil import parser  # arbitrary datetime strings -> datetime object
import datetime
#import argparse
import multiprocessing as mp
from cycler import cycler

from ECGFigure import ECGFigure

################################# Main Class: ##################################

class ECGClock(object):
    def __init__(self, title=None,
                 min_rad=0, max_rad=2000,
                 color_cycle=['b', 'm', 'c', 'k', 'y'],
                 parent_figure=None,
                 subplot=1):
        """Prepare a '24 hour' polar plot to add recordings to.  If no parent figure is
        specified, this will be a new standalone plot.  Otherwise, it will be a
        subplot on the parent figure.

        Keyword arguments:
        title -- title of this subplot (or whole figure, if this is the only subplot)
        min_rad -- inner radius of clock, in milliseconds
        max_rad -- outer radius of clock, in milliseconds
        color_cycle -- colors to cycle through when adding recordings/ranges.
                       e.g.: [plt.cm.autumn(i) for i in np.linspace(0, 1, 7)]
        parent_figure -- the matplotlib.figure.Figure that this clock will be on
        subplot -- position of this subplot on the parent figure
        """
        # Save pointers to important things, creating a new Figure if needed to hold this plot:
        if parent_figure:
            self.parent_figure = parent_figure
        else:
            self.parent_figure = ECGFigure()
        self.fig = self.parent_figure.fig
        self.subplot = subplot

        # Add this clock to the parent Figure:
        subplot_rows, subplot_cols = self.parent_figure.nrows, self.parent_figure.ncols
        self.ax = self.fig.add_subplot( subplot_rows, subplot_cols,
                                        self.subplot, projection='polar')
        #self.ax = self.ax.flatten()  # TODO: not needed?

        # Adjust axes parameters:
        self.ax.set_ylim(min_rad, max_rad)
        self.ax.set_theta_direction(-1)
        self.ax.set_theta_offset(np.pi/2.0)
        self.ax.set_xticklabels(['00:00', '03:00', '06:00', '09:00',
                                 '12:00', '15:00', '18:00', '21:00'])

        # Show time under mouse cursor (instead of angle):
        self.ax.format_coord = self.format_coord

        #self.ax.set_color_cycle(color_cycle)  # TODO?: overall vs subplot color cycle
        self.ax.set_prop_cycle( cycler('color', color_cycle) )
        
        if parent_figure:
            self.set_title(title)
        else:
            self.parent_figure.set_title(title)

    def set_title(self, title):
        """Set/change the title for this plot.

        Keyword arguments:
        title -- the new title (string)
        """
        if title:
            self.ax.set_title(title + '\n')
            # \n to prevent overlap with '00:00' label

    def add_recording(self, filename, label=None, color=None, filtering=0,
                      time_col=0, val_col=1):
        """Read dataset from file and add it to the plot.  If there are large gaps
        in time, data points in between will be interpolated.  Note: it may take
        about 10 seconds for this function to run on 24-hour beat-to-beat data.

        Keyword arguments:
        filename -- csv to read data from
        label -- what to call this on the plot legend
        color -- line color (or None to follow normal rotation)
        filtering -- width of filter in minutes, or 0 to disable filtering
        time_col -- column in csv containing time strings, 0-indexed
        val_col -- column in csv containing feature values, 0-indexed
        """
        # TODO: could default None label to filename
        times, values = load_from_csv(filename, cols_wanted=[time_col,val_col])
        values = sec_to_msec(values)  # TODO: allow bypassing this
        angles = times_to_angles(times)
        if filtering:
            values = medfilt( times, values, filter_width=filtering )

        interp_angles, interp_values = polar_interp( angles, values )  # TODO: pass dTheta too
        self.ax.plot(interp_angles, interp_values, zorder=0, color=color, label=label)
        # TODO: note/handle different starting dates when multiple recordings are added.

    def add_annotation(self, time, r, x, y, label='', color='black'):
        """Add an annotation to a plot.  The annotation consists of a point at (time,r)
        and an arrow from (x,y) to that point.  The label appears at the tail of the arrow.

        Keyword arguments:
        time, r -- marker location
        x, y -- text location (fraction from bottom left of ENTIRE FIGURE)
        label -- text at arrow tail
        color -- line and marker color
        """
        if (x <= 0.5):
            ha = 'left'
        else:
            ha = 'right'
        if (y <= 0.5):
            va = 'bottom'
        else:
            va = 'top'

        th = times_to_angles([time], parallel=False)[0]

        #print self.get_ax(subplot)  # debugging

        self.ax.plot(th, r, 'o', color=color, mew=0)
        self.ax.annotate(label,
                         xy=(th, r),
                         xytext=(x, y),
                         color=color,
                         textcoords='figure fraction',  # TODO: is subplot fraction an option?
                         arrowprops=dict(facecolor=color, ec=color, shrink=0.05,
                                         width=1, headwidth=8),
                         horizontalalignment=ha,
                         verticalalignment=va
        )

    def add_percentile_range(self, filename, lower=25, upper=75,
                             label=None, color=None, alpha=0.2, 
                             smoothing=20):
        """Load a precomputed range from a file, and add it to the plot.  zorder is
        set to -1 in this function, so foreground items should use zorder>-1.
        We assume that the axis has theta_direction=-1 and theta_offset=pi/2.

        Keyword arguments:
        filename -- csv to read data from.  columns should be {time, 0%, 1%, ... , 100%}
        lower -- lower percentile bound to show
        upper -- upper percentile bound to show
        label -- what to call this region on the plot legend
        color -- color of the new region (note: you should probably specify this... see
                 http://stackoverflow.com/questions/30535442/matplotlib-fill-between-does-not-cycle-through-colours)
        alpha -- transparency of the new region
        smoothing -- median filter window size for smoothing lower and upper bounds
        """
        # Load file:
        times, lower_bounds, upper_bounds = load_from_csv(filename,
                                                           cols_wanted=[0,lower+1,upper+1],
                                                           col_fmt=[str,float,float])
        # TODO?: allow interpolation between columns, e.g. to get 2.5 percentile
        thetas = times_to_angles( times )
        lower_bounds = sec_to_msec(lower_bounds)
        upper_bounds = sec_to_msec(upper_bounds)

        if smoothing:
            # smooth lower and upper bounds using medfilt().
            lower_bounds = medfilt( times, lower_bounds, filter_width=smoothing )
            upper_bounds = medfilt( times, upper_bounds, filter_width=smoothing )
        # (TODO?: may want to do that after the next block)
        # TODO: average may be nicer than median for this

        if ( np.mod(thetas[-1], 2*np.pi) != np.mod(thetas[0], 2*np.pi) ):
            # if the region doesn't end at the same angle where it started, add
            # one more point to close the area
            thetas += [ thetas[0] ]
            while (thetas[-1] < thetas[-2]):
                # ensure thetas[-1] > thetas[-2]:
                thetas[-1] += 2*np.pi
            lower_bounds += [ lower_bounds[0] ]
            upper_bounds += [ upper_bounds[0] ]

        # TODO: interpolate more data points?

        # Plot:
        self.ax.fill_between(thetas, lower_bounds, upper_bounds,
                             # alpha=alpha, linewidth=0, zorder=-1,
                             alpha=alpha, linewidth=0.001, zorder=-1,  # lw=0 is bugged in some versions of mpl
                             label=label, color=color)

    def add_legend(self):
        """Add the legend to the top right of the figure, outside of the plot area.
        """
        self.ax.legend(loc="upper left", bbox_to_anchor=(1,1.1))
        # TODO: maybe pass other args through to ax.legend()
        # TODO: overall vs subplot legends?

    def show(self):
        """Show the figure on screen, i.e. with all subplots including this one.
        """
        self.parent_figure.show()
        # TODO: only show individual subplot here... update description then.

    def save(self, filename):
        """Save the figure to disk, i.e. with all subplots including this one.  If the
        plot has been modified (zoomed, resized, etc.) via show(), these changes
        will be included.

        Keyword arguments:
        filename -- file to save to, e.g. 'qt_clock.png'
        """
        self.fig.savefig(filename, bbox_inches='tight')
        # TODO: default to title as filename if none specified?
        # TODO: only save individual subplot here... update description then.

    def format_coord(self, th, r):
        """Return a human readable string from a (theta, r) coordinate."""
        return 'time=' + angle_to_time(th) + ', val=%1.0f'%(r)  # TODO: units on val?

# # TODO: kwargs in several places?

############################### Extra Functions: ###############################

def angle_to_time(th):
    """Convert an angle like pi/2 into a string like '06:00'.

    Keyword arguments:
    th -- angle in radians, starting from 0 at 00:00 and increasing to 2pi at 24:00
    """
    th = np.mod(th, 2*np.pi)
    minute = 1.0*th/(2*np.pi) * 24  # well, hour not minute
    hour   = int(np.floor(minute))
    minute = int(round( (minute - hour) * 60 ))
    return str(hour).zfill(2) + ":" + str(minute).zfill(2)
    # TODO: maybe use this function to generate x tick labels?

def load_from_csv(csv_filename, cols_wanted=[0,1], col_fmt=[str,float]):
    """Read a CSV file and return the specified columns as separate lists, converted
    to the specified data types.  Using the default values of cols_wanted and
    col_fmt, for example, we could load a CSV where each line was in the format
    "11:06:30,452".  If we can't parse the first row of the csv file, we will
    assume it was a header row and skip it.

    Keyword arguments:
    csv_filename -- the file to read from
    cols_wanted -- which columns to read.  return values will follow this order.
    col_fmt -- respective data types for columns listed in cols_wanted.
    """
    if ( len(cols_wanted) != len(col_fmt) ):
        raise IndexError('load_from_csv(): cols_wanted and col_fmt must be same length.')

    results = [ [] for i in range(len( cols_wanted )) ]
    with open(csv_filename, 'rtU') as csvfile:  # TODO: fix Windows bug; it doesn't like 'U'.  (deprecated?)
        csv_reader = csv.reader(csvfile)
        for i, row in enumerate(csv_reader):
            new_results = ['' for _ in range(len( cols_wanted ))]
            try:
                for j, col in enumerate(cols_wanted):
                    new_results[j] = col_fmt[j]( row[col] )
                    # TODO: handle NaN/blank values somehow... probably not in here, though.
            except ValueError:
                if (i == 0):
                    continue  # assume we just choked on header row
                else:
                    raise
            for j, res in enumerate(new_results):
                results[j].append( res )
    return tuple(results)

def times_to_angles(times, parallel=True):
    """Convert a list of times (where each time is a string like
    '1998-04-02T09:26:03.620') into angles in radians.  Midnight is 0 (or 2pi),
    6AM is pi/2, noon is pi, 6PM is 3pi/2.  These values can be mapped to the
    proper clock positions (i.e. midnight at 'top' of clock) using
    theta_direction=-1 and theta_offset=pi/2 on your axes.  If the date is
    available in the time strings, it will be used to 'wrap around' the clock,
    e.g. angles for times tomorrow will be 2pi higher than those for today.  If
    dates are unknown/unlisted, we will always assume that times wrapping
    through midnight proceed to the next consecutive day - e.g. if a data point
    at '23:59' is followed by one at '00:01', we assume 2 minutes elapsed even
    though it could really be 2 minutes plus 24*k hours.

    Keyword arguments:
    times -- times, as a list of strings.  e.g.: ['11:06:15', '11:06:20']
    parallel -- should we parallelize the conversion process?
    """
    # angles will be indexed from the midnight before the data started:
    first_time = parser.parse(times[0]).replace(hour=0, minute=0, second=0, microsecond=0)

    running_offset = 0  # for wrapping around when we don't know dates

    #times = np.array(times)

    # parser.parse() is quite slow, so we usually want to parallelize this part:
    if parallel:
        pool = mp.Pool(processes=mp.cpu_count())
        times = pool.map(parser.parse, times)
        pool.close()
        pool.join()
    else:
        times = [parser.parse(t) for t in times]
    # (Parsing could be done much faster if we could assume a standard input
    # format, but we want to accept fairly arbitrary date/time strings.)

    angles = []
    for t in times:
        h = t.hour; m = t.minute; s = t.second; us = t.microsecond
        # year = t.year; month = t.month; day = t.day
        t_as_hr = h + ( (((s + (us/1e6))/60.0) + m) / 60.0 )  # time as hour of day
        angle = (t_as_hr/24.0) * 2*np.pi  # time in radians

        # If we know dates:
        angle += 2*np.pi * (t - first_time).days  # offset by 2pi per day elapsed

        # If we don't know dates:
        angle += running_offset
        if ( (angles) and (angle < angles[-1]) ):
            running_offset += 2*np.pi
            angle += 2*np.pi

        angles.append(angle)
    return angles

def polar_interp( thetas, rs, min_dTheta=(2*np.pi/1440) ):
    """Add more points to theta and r vectors to fill in large gaps in theta.  This
    is needed because matplotlib draws straight lines between data points, even
    on polar axes.

    Keyword arguments:
    thetas -- list of theta values, same length as rs.  values must be consecutive and increasing.
    rs -- list of radial values, same length as thetas
    min_dTheta -- minimum output resolution, default value is 1 point per minute (1440 mins/360 degrees)
    """
    th_new = [ thetas[0] ]
    r_new = [ rs[0] ]

    for i in range(1,len(thetas)):

        dTheta = thetas[i] - thetas[i-1]

        if ( dTheta > min_dTheta ):
            # poor resolution here, add more points (TODO: simplify this part)
            points_to_add = int(np.ceil(1.0*dTheta/min_dTheta - 1))
            new_dth = 1.0 * dTheta / (points_to_add+1)
            th_to_add = [ thetas[i-1] + pt*new_dth for pt in range(1, points_to_add+1) ]
            r_to_add = np.interp(th_to_add,
                                 [ thetas[i-1], thetas[i] ],
                                 [ rs[i-1], rs[i] ] )
            th_new.extend(th_to_add)
            r_new.extend(r_to_add)

        # Always keep the existing point:
        th_new.append(thetas[i])
        r_new.append(rs[i])

    return th_new, r_new
    # TODO: use this function for loaded ranges too, not just single-patient data

def general_filter( times, values, filter_width=5, filt_type=np.median ):
    """Filters the values list with a width of filter_width minutes.  Returns
    the filtered values list.  Note that the results at the beginning and end of
    the list will be skewed; we don't pad values 'outside' of the list.

    Keyword arguments:
    times -- list of time strings
    values -- list of values corresponding to times
    filter_width -- in minutes
    filt_type -- the function to apply to each window, e.g. max or np.average.
    """
    assert (len(times) == len(values))
    angles = times_to_angles(times)  # because this handles time wraparound for
                                     # us, and allows us to compare floats
                                     # rather than time strings/objects
    # TODO: don't duplicate work; allow passing in angles if we already have them.
    filter_width_radians = 1.0 * filter_width / (24*60) * 2*np.pi
    values_filtered = values[:]
    values = np.array(values)  # for small speedup
    angles = np.array(angles)  # for big speedup

    # Single-threaded:
    for i in range( len(values) ):
        x = np.searchsorted(angles, angles[i] - filter_width_radians/2.0              )  # start index
        y = np.searchsorted(angles, angles[i] + filter_width_radians/2.0, side='right')  # end index
        values_filtered[i] = filt_type( values[x:y] )  # TODO?: optimize this if filter gets wide
        # TODO?: smarter/faster search, e.g. by remembering where we left off
        # last time or by parallelizing that for loop

    # Multi-"threaded":
    # pool = mp.Pool(processes=mp.cpu_count())
    # fargs = [(angles, values, filter_width_radians, i)
    #          for i in range(len(angles))]  # TODO: avoid duplication
    # values_filtered = pool.map(par_med_filt, fargs)  # TODO
    # pool.close()
    # pool.join()
    # Doesn't save much time, probably because of duplication.

    return values_filtered

def medfilt( times, values, filter_width=5 ):
    """Median-filters the values list with a width of filter_width minutes.  Returns
    the filtered values list.  Note that the results at the beginning and end of
    the list will be skewed; we don't pad values 'outside' of the list.

    Keyword arguments:
    times -- list of time strings
    values -- list of values corresponding to times
    filter_width -- in minutes
    """
    return general_filter( times, values, filter_width=filter_width, filt_type=np.median )

def par_med_filt(packed_args):
    """Computes the output of a median filter at position i in values, where the
    window boundaries are determined by angles and filter_width_radians.  This
    is intended to be used as a helper for medfilt().

    Keyword arguments:
    packed_args -- (list of angles, list of values, filter_width_radians, position)
    """
    angles, values, filter_width_radians, i = packed_args
    assert (len(angles) == len(values))
    x = np.searchsorted(angles, angles[i] - filter_width_radians/2.0              )  # start index
    y = np.searchsorted(angles, angles[i] + filter_width_radians/2.0, side='right')  # end index
    return np.median( values[x:y] )

def decimate( times, values, q=5, aggregator=np.average, parallel=True):
    """Downsamples a set of times and values.  The final chunk will be discarded if
    it doesn't contain q samples.  The aggregator function is applied
    independently to the times and the values.  Example:
    decimate( ['1:35','1:36','1:37','1:38'], [4,3,2,1], q=2, aggregator=max)
    returns
    ( ['1:36','1:38'], [4, 2] ).
    Note that the behavior of this function may not be what you intended if the
    spacing of the data points is irregular.

    Keyword arguments:
    times -- list of time strings
    values -- list of values corresponding to times
    q -- downsampling factor
    aggregator -- method used to aggregate each group of samples
    parallel -- parse input times in parallel
    """
    # TODO: allow time-based window rather than point-based.
    # TODO?: just act on one vector of values, let person run it again to do
    # times?  need to know if we have to do time conversions though.  can we
    # average, etc. normal datetime objects?  or what about user selecting 2
    # different aggregators?
    assert (len(times) == len(values))

    times_downsampled = [None]*(len(times)//q)
    values_downsampled = [None]*(len(values)//q)

    if parallel:
        pool = mp.Pool(processes=mp.cpu_count())
        times = pool.map(parser.parse, times)
        pool.close()
        pool.join()
    else:
        times = [parser.parse(t) for t in times]

    # WIP:
    times = [t.timestamp() for t in times]  # datetime objects -> unix time (float)
    for i in range( len(values)//q ):
        values_downsampled[i] = aggregator( values[i*q:(i+1)*q] )
        times_downsampled[i] = aggregator( times[i*q:(i+1)*q] )
    times_downsampled = [datetime.datetime.fromtimestamp(t).isoformat() for t in times_downsampled]  # unix time -> time string
    # Note: python3 required...

    return times_downsampled, values_downsampled

def sec_to_msec(values, cutoff=10):
    """Take a list of values and return either: (a) the same list or (b) the list
    multiplied by 1000.  If the values in the list are 'too low', we assume we
    must do (b).  This is useful when we read a bunch of times that were
    supposed to be in milliseconds, but they might have been in seconds by accident.

    Keyword arguments:
    values -- list of values to be converted
    cutoff -- if most values are above this number, we won't convert the list
    """
    if (np.median(values) < cutoff):
        values = [val * 1000.0 for val in values]
    return values

def msec_to_sec(values, cutoff=10):
    """Take a list of values and return either: (a) the same list or (b) the list
    divided by 1000.  If the values in the list are 'too high', we assume we
    must do (b).  This is useful when we read a bunch of times that were
    supposed to be in seconds, but they might have been in msec by accident.

    Keyword arguments:
    values -- list of values to be converted
    cutoff -- if most values are below this number, we won't convert the list
    """
    if (np.median(values) > cutoff):
        values = [val / 1000.0 for val in values]
        # values = np.array(values)
        # values = values / 1000.0
        # values = values.tolist()
    return values

################################### main(): ####################################

if __name__=='__main__':
    #mp.freeze_support()  # only needed in Windows
    pass

#################################### TODO: #####################################

# - split into different subclasses and eliminate duplicate code
# - finish updating sample range CSVs
# - fix spacing between figures, titles, etc. when using subplots
# - convert normal range csv files to msec
# - parse/convert times ONCE rather than repeatedly?  maybe have a Recording
#   object?
# - start/end arrows (or other markers)?  could get messy... maybe just an
#   option to list the start/end times
# - adding / generating ranges from sets of patients
# - make some things "private"
# - keep docstrings and argparse help updated
# - specify starting offset of a plot, e.g. where '00:00' in csv really means
#   some other time?
# - alpha setting (e.g. for all the 0.2 bg values)
# - GUI?
# - store list of rows rather than separate columns?
# - could try to handle string entries like '0.5s' or '450ms' etc.
#   instead of requiring float.

################################################################################
