# lambdata-davidanagy

This is a practice repository for my Lambda School assignment (Unit 3 Sprint 1).

This module includes functions designed by me to help with data science-related coding.

[Here's a link](https://test.pypi.org/project/lambdata-davidanagy/0.1.6/) to the current version, if you want to import my functions and use them.

# Features

This module currently contains [five functions/methods](https://github.com/davidanagy/lambdata-davidanagy/blob/master/lambdata_davidanagy/df_utils.py), three related to manipulating datasets and two relating to statistics.

## Dataset functions

### check_nulls

The check_nulls method works similarly to the pandas [isnull method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.isnull.html) combined with .sum(), but with two key differences to increase readability. First, it skips over columns that contain no missing values so you only see the ones that have them (especially useful for datasets with many columns). Second, it outputs the total number of missing values at the end. For example, take the following dataframe:

zero_nan | one_nan | all_nan
--- | --- | ---
1 | np.nan | np.nan
2 | 2 | np.nan
3 | 3 | np.nan

Here's how we sum up the missing values with pandas:

```python
import pandas as pd
df.isnull().sum()
```

Which gives us:

```
zero_nan    0
one_nan     1
all_nan     3
```

On the other hand, my check_nulls function:

```python
from lambdata_davidanagy.df_utils import Dataframe_funcs
func = Dataframe_funcs()
func.check_nulls(df)
```

Results in:

```
Column "one_nan" contains 1 missing value.
Column "all_nan" contains 3 missing values.
This dataset contains 4 missing values.
```

### add_list_to_df

This function streamlines the process of converting a list to an array and adding it to a dataframe as a column. Note that, if you don't specify a name for the new column, it'll default to "list."

### split_date

This method takes a column with dates on it, converts it to datetime using the [pandas to_datetime function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html), then splits it into years, months, and days, each receiving its own column.

The split_date function has the same hyperparameters as the pandas to_datetime function, except for "Drop"; if you would like to delete the original dates column, set Drop=True. (Though be careful to also specify *both* the dataframe *and* the column with the dates when you call the method.)

## Statistics functions

### conf_mat

This creates a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix), just like the [sklearn command](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html), except it includes "Predicted" and "Actual" row/column labels for clarity and readability.

### train_val_test_split

This is similar to the [sklearn train_test_split function](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), except it provides a *tripartite* split [with a validation set](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets#Validation_dataset). If left unspecified, the train/val/test ratio will be 0.7/0.2/0.1. Note that there are also Shuffle and Partition parameteres, same as the sklearn function.

# Dependencies

* [NumPy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)

# License

[MIT License](https://github.com/davidanagy/lambdata-davidanagy/blob/master/LICENSE)

# Contact

Please email me at [davidanagy@gmail.com](mailto:davidanagy@gmail.com) if you have any comments, questions, or suggestions.

Also, [visit my website](https://davidanagy.github.io/)!