"""
lambdata - a collection of data science helper functions
""" 
import pandas as pd
import numpy as np

#sample code

#sample datasets
ONES = pd.DataFrame(np.ones(10))
ZEROS = pd.DataFrame(np.zeros(50))

#sample functions
def inc(x):
    return x+1