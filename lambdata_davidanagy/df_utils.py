"""
utility functions for working with pandas dataframes
"""
import pandas as pd
import numpy as np

TEST_DF1 = pd.DataFrame({'ones': [1] * 10, 'twos': [2] * 10})
TEST_DF2 = pd.DataFrame(
    {'dates': ['1/1/2000'] * 10,
    'one_nan': [2] * 9 + [np.nan],
    'ten_nans': [np.nan] * 10})

TEST_LIST1 = [3] * 10
TEST_LIST2 = [8,3,5,3,3,5,5,8,3,5]

class Dataframe_funcs:

    """
    Functions related to manipulating dataframes
    """

    def __init__(self):
        pass

    def check_nulls(self, X):
        '''
        Checks a dataframe for nulls and reports them in a "pretty" format
        '''
        if type(X) != pd.core.frame.DataFrame:
            print('ERROR: Input must be a Pandas dataframe')
        else:
            columns = X.columns
            nulls = 0  # This is to keep track of total nulls.
            for i in range(len(columns)):
                nan_num = X.isnull().sum()[i]
                if nan_num == 0:
                    continue
                    # This way you don't print out every column,
                    # just the ones that contain nulls.
                elif nan_num == 1:
                    print(f'Column "{columns[i]}" contains 1 missing value.')
                    # Separating this out so I get the singular noun.
                    nulls += 1
                else:
                    print(f'Column "{columns[i]}" contains {nan_num} missing values.')
                    nulls += nan_num
            if nulls == 0:
                print('This dataset contains no missing values!')
            elif nulls == 1:
                print('This dataset contains 1 missing value.')
            else:
                print(f'This dataset contains {nulls} missing values.')

    def add_list_to_df(self, list_, df, name='list'):
        # (Using 'list_' to avoid overwriting 'list.')
        """Adds list to dataframe as a new column"""
        if type(list_) != list:
            print("ERROR: First input must be a list")
        elif type(df) != pd.core.frame.DataFrame:
            print("ERROR: Second input must be a Pandas dataframe")
        else:
            column = pd.DataFrame(list_, columns=[name])
            if len(column) != len(df):
                print('WARNING: List and dataframe have different lengths.')
                print('This function will create missing values.')
            return pd.concat([df, column], axis=1)

    def split_date(
            self, df, column, drop=False, errors='raise',
            dayfirst=False, yearfirst=False,
            utc=None, format=None, exact=True, unit=None,
            infer_datetime_format=False,
            origin='unix', cache=True):
        """Function to split dates into multiple columns"""
        idf = infer_datetime_format

        if type(df) != pd.core.frame.DataFrame:
            print("ERROR: First parameter must be a Pandas dataframe.")
        else:
            X = df.copy()
            X['split'] = pd.to_datetime(X[column], errors=errors,
                                        dayfirst=dayfirst,
                                        yearfirst=yearfirst,
                                        utc=utc, format=format,
                                        exact=exact, unit=unit,
                                        infer_datetime_format=idf,
                                        origin=origin, cache=cache)
            # Function creates a new datetime column,
            # then deletes the new column.
            # This is so user has a choice of whether
            # or not to convert the original column
            # to datetime format. (Alternately they can
            # just delete the column by setting "drop=True.")
            X[f'{column}_year'] = X['split'].dt.year
            X[f'{column}_month'] = X['split'].dt.month
            X[f'{column}_day'] = X['split'].dt.day
            X = X.drop('split', axis=1)
            if drop:
                X = X.drop(column, axis=1)
            return X


class Stats_funcs:

    """
    Functions related to statistics/modeling
    """

    def conf_mat(self, y_true, y_pred, labels='auto'):
        """Returns a confusion matrix, with labels for easier interpretation"""
        if type(y_true) != np.ndarray:
            print('ERROR: Inputs must be NumPy arrays.')
        elif type(y_pred) != np.ndarray:
            print('ERROR: Inputs must be NumPy arrays.')
        elif len(y_true) != len(y_pred):
            print('ERROR: The arrays must have the same length.')
        else:
            if labels == 'auto':
                labels = []
                for label in np.unique(y_true):
                    # i.e. for each unique value in y_true
                    labels.append(label)
                for label in np.unique(y_pred):
                    labels.append(label)
                labels = list(set(labels))
                # Putting it into a set first
                # so I only have each unique label once.
            rows = []
            for k in range(len(labels)):  # For each label,
                row = [0] * len(labels)
                # create a row with with length
                # equal to the number of unique labels.
                for m in range(len(y_pred)):  # For each value in y_pred,
                    if y_pred[m] == labels[k]:
                        # if that value is equal to the label in question,
                        for n in range(len(labels)):
                            if labels[n] == y_true[m]:
                                # then take the label that value is equal to,
                                row[n] += 1
                                # and add 1 to the corresponding row location.
                rows.append(row)
            row_labels = [f'Predicted {label}' for label in labels]
            column_labels = [f'Actual {label}' for label in labels]
            return pd.DataFrame(data=rows,
                                index=row_labels,
                                columns=column_labels)

    def train_val_test_split(
            self, X, sizes='auto',
            random_state=np.random.RandomState(),
            shuffle=True, stratify=None):
        """Applies train/validate/test split to a dataframe"""
        if sizes == 'auto':
            train_size = 0.7
            val_size = 0.2
            test_size = 0.1
        elif type(sizes) != list:
            print("ERROR: 'sizes' must either be \
                a list of 3 numbers or 'auto.'")
        elif len(sizes) != 3:
            print("ERROR: 'sizes' must either be \
                a list of 3 numbers or 'auto.'")
        elif sum(sizes) != 1:
            print("ERROR: 'sizes' must add up to exactly 1.")
        else:
            train_size = sizes[0]
            val_size = sizes[1]
            test_size = sizes[2]

        if shuffle:
            if stratify is None:
                Y = X.sample(frac=1, random_state=random_state)
                Y = Y.reset_index()
                # Resetting the index so the .loc command works properly.
                test_rows = np.round(len(Y) * test_size) - 1
                # Subtracting by one because I'm going to be running .loc
                # over Y's index, and the index starts at 0.
                if test_rows == -1:
                    test_rows == 0
                # Setting test_rows and val_rows first
                # to make sure neither equals -1 in small datasets.
                val_rows = np.round(len(Y) * val_size)
                if val_rows == 0:
                    val_rows == 1
                # Now that I'm past the first row, we're not starting
                # at zero, so there's no need to subtract 1 from val_rows.
                test = Y.loc[0:test_rows].set_index('index')
                # Putting back the original index.
                # User then has the choice to use reset_index themselves.
                del test.index.name
                # There is no need to name the index "index."
                val = Y.loc[test_rows+1:test_rows+val_rows].set_index('index')
                # Starting where the test set ended,
                # ending when you reach val_rows total rows.
                del val.index.name
                train = Y.loc[test_rows+val_rows+1:len(Y)].set_index('index')
                # Starting where the val set ended.
                del train.index.name
            else:
                # This is for when user wants the train/val/test sets
                # to have the same proportion of values in
                # a certain column as the original dataset.
                counts = X[stratify].value_counts()
                train = pd.DataFrame()
                val = pd.DataFrame()
                test = pd.DataFrame()
                for value in range(len(counts)):
                    temp = X[X[stratify] == value]
                    # Isolate rows where the stratification column
                    # equals a certain value.
                    temp2 = temp.sample(frac=1, random_state=random_state)
                    temp2 = temp2.reset_index()
                    test_rows = np.round(len(temp2) * test_size) - 1
                    if test_rows == -1:
                        test_rows == 0
                    val_rows = np.round(len(temp2) * val_size)
                    if val_rows == 0:
                        val_rows == 1
                    temp_to_test = temp2.loc[0:test_rows]
                    temp_to_test = temp_to_test.set_index('index')
                    del temp_to_test.index.name
                    temp_to_val = temp2.loc[test_rows+1:test_rows+val_rows]
                    temp_to_val = temp_to_val.set_index('index')
                    del temp_to_val.index.name
                    temp_to_train = temp2.loc[test_rows+val_rows+1:len(temp2)]
                    temp_to_train = temp_to_train.set_index('index')
                    del temp_to_train.index.name
                    train = pd.concat([train, temp_to_train])
                    # Concatenate on each step in the loop
                    # so we end up with the entire dataset
                    # after the loop finishes.
                    val = pd.concat([val, temp_to_val])
                    test = pd.concat([test, temp_to_test])
        else:
            # For when user doesn't want the split to occur randomly.
            if stratify is not None:
                print("ERROR: If shuffle = 'False,' \
                    then stratify must = 'None.'")
            else:
                test_rows = np.round(len(X) * test_size)
                if test_rows == 0:
                    test_rows == 1
                # This time I'm going to do .loc starting with
                # the train rows, so train_rows will be the value
                # that I have to subtract 1 from.
                val_rows = np.round(len(X) * val_size)
                if val_rows == 0:
                    val_rows == 1
                train_rows = len(X) - (test_rows + val_rows) - 1
                train = X.loc[0:train_rows]
                # Doing train first this time because
                # that's more intuitive when there's no shuffling.
                val = X.loc[train_rows+1:train_rows+val_rows]
                test = X.loc[train_rows+val_rows+1:len(X)]

        return train, val, test
