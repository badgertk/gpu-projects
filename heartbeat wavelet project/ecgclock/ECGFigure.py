#!/usr/bin/env python

"""TODO.
"""

################################### Imports: ###################################

# Need to do this to prevent warnings/errors when saving a batch of clocks:
import matplotlib
matplotlib.use('Qt4Agg')  # adds improved zoom over default backend
# matplotlib.use('GTKCairo')  # switch to vector graphics
# matplotlib.use('GTKAgg')  # nothing fancy over default.  similar to WXAgg.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

################################# Main Class: ##################################

class ECGFigure():
    """TODO.  It would make sense for this to be a subclass of
    matplotlib.figure.Figure, but ...
    """

    def __init__(self, nrows=1, ncols=1, title=None, inches_per_row=7, inches_per_col=7):
        """
        Create a new figure with enough space to hold nrows*ncols subplots.

        Keyword arguments:
        title -- the new title (string)
        nrows, ncols -- TODO
        inches_per_row, inches_per_col -- TODO
        """
        # Store these for easy lookup later, rather than parsing
        # fig._axstack... sadly, this is basically the purpose of this class:
        self.nrows = nrows
        self.ncols = ncols

        # Create the figure:
        self.fig = plt.figure( figsize=(ncols*inches_per_col,
                                        nrows*inches_per_row))
        # TODO?: squeeze=False
        
        # Adjust spacing:
        self.fig.subplots_adjust(  wspace=0.35, hspace=0.2,
                                   top=np.exp(-(nrows - 1)/8.0)*(0.9-1)+1  )  # empirical adjustment of top spacing
        #self.fig.tight_layout()

        self.set_title(title)

        # TODO: get subplot titles as arg too?  then remember:
        #     assert(  (subplot_titles==None) or ( len(subplot_titles) == subplot_rows*subplot_cols )  )

    def __del__(self):
        """Free memory by closing the figure."""
        try:
            plt.close(self.fig.number)
        except AttributeError:
            pass

    def set_title(self, title):
        """Set/change the title for this figure.

        Keyword arguments:
        title -- the new title (string)
        """
        if title:
            self.fig.suptitle(title, fontsize=16)  # TODO: \n needed?

    def show(self):
        """Show the figure on screen.  Note that the current figure will be set to this
        one.
        """
        # self.fig.subplots_adjust(right=0.7)  # TODO: adjust to make legend(s) visible
        plt.figure(self.fig.number)  # set current figure to ours
        plt.show()  # TODO: block? (should be param?)
        # self.fig.show()  # alternate method, which closes immediately :/
        # TODO: test when we have many figures... does it show all of them, or
        # just the one we wanted?

    def save(self, filename):
        """Save the figure to disk, i.e. with all subplots.  If the plot has been
        modified (zoomed, resized, etc.) via show(), these changes will be
        included.

        Keyword arguments:
        filename -- file to save to, e.g. 'qt_clock.png'
        """
        self.fig.savefig(filename, bbox_inches='tight')

#     def get_ax(self, subplot=None):
#         """Returns the current axes object, or the axes object of subplot n.

#         Keyword arguments:
#         subplot -- which subplot to get axes of (1-indexed)
#         """
#         if subplot:
#             return self.ax[subplot-1]
#         else:
#             return plt.gca()

#     def set_ax(self, subplot=None):
#         """Set/change the current axes object.

#         Keyword arguments:
#         subplot -- which subplot to select (1-indexed)
#         """
#         if subplot:
#             #plt.figure(self.fig.number)  # set current figure to ours
#             plt.sca( self.ax[subplot-1] )
#         else:
#             pass

#################################### TODO: #####################################

# ...

################################################################################
