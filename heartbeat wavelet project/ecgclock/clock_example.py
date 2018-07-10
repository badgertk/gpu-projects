#!/usr/bin/env python

from QTClock import QTClock
from ECGFigure import ECGFigure  # only needed for subplots example
import multiprocessing as mp  # only needed for Windows

############################### Single example: ###############################

def single_example():
    my_clock = QTClock('QTcB: Baseline vs. Drug Trial')
    
    # Add two recordings to the plot.  Filtering is disabled because this data was
    # already filtered.
    my_clock.add_recording('./example_data/baseline_eg.csv', label='baseline')
    my_clock.add_recording('./example_data/drug_eg.csv',     label='drug'    )
    
    # Annotate an interesting point on this plot.
    my_clock.add_annotation('13:03', 476, 0.1, 0.05,
                            label='476ms @ 13:03')
    
    # Show two 'tiers' of healthy QTc ranges: IQR in darker green, and a wider
    # percentile range in lighter green around it.  IQR is darker because the
    # regions overlap.  We only label one of them, because we don't want redundant
    # entries in the legend.
    my_clock.add_percentile_range('./normal_ranges/QTcB_healthy_male.csv',
                                  lower=7, upper=93, color='g', alpha=0.15, label='healthy')
    my_clock.add_percentile_range('./normal_ranges/QTcB_healthy_male.csv',
                                  lower=25, upper=75, color='g', alpha=0.15)
    
    # >500ms will be highlighted red:
    my_clock.add_danger_range(500)
    
    my_clock.add_legend()
    
    my_clock.save('baseline_vs_drug.png')
    #my_clock.show()  # view it in an interactive window

############################## Subplots example: ##############################

def subplots_example():
    my_fig = ECGFigure(nrows=1, ncols=2, title='QTcB: Baseline vs. Drug Trial')

    before_clock = QTClock('Baseline', parent_figure=my_fig, subplot=1)
    after_clock =  QTClock('Drug',     parent_figure=my_fig, subplot=2)
    
    # Add recordings to the two separate subplots.  Again, filtering is disabled
    # because this data was already filtered.
    before_clock.add_recording('./example_data/baseline_eg.csv')
    after_clock.add_recording('./example_data/drug_eg.csv')
    
    # Annotate the same point as before, but with new (x,y) coordinate due to
    # subplots.
    after_clock.add_annotation('13:03', 476, 0.6, 0.1,
                               label='476ms @ 13:03')
    
    # Add static green and red bands to indicate typical healthy and dangerous QTc
    # regions:
    before_clock.add_default_ranges()
    after_clock.add_default_ranges()
    
    my_fig.save('baseline_vs_drug_subplots.png')
    #my_fig.show()  # view it in an interactive window

################################ Main program: ################################

if __name__=='__main__':
    mp.freeze_support()  # only needed in Windows

    single_example()
    subplots_example()

################################################################################
