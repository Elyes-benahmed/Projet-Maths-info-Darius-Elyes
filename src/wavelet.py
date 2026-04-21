# -*- Encoding: Latin-1 -*-
#!/usr/bin/python

import sys, time
from os.path import expanduser
home = expanduser("~")

import numpy as np
import pywt


class wavelet:

    # -----------------------------------
    # Class constructor	
    # -----------------------------------
    def __init__(self, wavename):

        self.wavename = wavename	
        self.set_filters()

    # -----------------------------------
    # Creates the filters associated to the wavelet	
    # -----------------------------------		
    def set_filters(self):

        w = pywt.Wavelet(self.wavename)
        self.dec_high = w.dec_hi
        self.dec_low = w.dec_lo
        self.rec_high = w.rec_hi
        self.rec_low = w.rec_lo	




