"""Testing file for lambdata functions"""

import unittest
from df_utils import Dataframe_funcs, Stats_funcs
import pandas as pd
import numpy as np

df1 = pd.DataFrame({'ones': [1] * 10, 'twos': [2] * 10})
list1 = [3] * 10
df2 = pd.DataFrame({'ones': [1] * 10, 'twos': [2] * 10,
                    'threes': [3] * 10})
df3 = df2.copy()
df3.columns = ['ones', 'twos', 'list']

# For testing train/val/test split, I use the following auto ratios:
# train = 0.7, val = 0.2, test = 0.1
df1_train = df1.loc[0:6]
df1_val = df1.loc[7:8]
df1_test = pd.DataFrame(df1.loc[9]).T
# Need to do this for df1_test or it'll just be an array.

func1 = Dataframe_funcs()
func2 = Stats_funcs()

class DF_Tests(unittest.TestCase):
    """tests for dataframe functions"""
    def test_addlist_name(self):
        result = func1.add_list_to_df(list1, df1, 'threes')
        self.assertTrue(df2.equals(result))

    def test_addlist_default(self):
        result = func1.add_list_to_df(list1, df1)
        self.assertTrue(df3.equals(result))

class Stat_Tests(unittest.TestCase):
    """tests for stats functions"""
    def test_3way_split(self):
        results = func2.train_val_test_split(df1, shuffle=False)
        result_train = results[0]
        result_val = results[1]
        result_test = results[2]
        self.assertTrue(df1_train.equals(result_train) &
                        df1_val.equals(result_val) &
                        df1_test.equals(result_test))

if __name__ == '__main__':
    unittest.main()
