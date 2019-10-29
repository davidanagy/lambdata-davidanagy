"""
utility functions for working with pandas dataframes
""" 
import pandas as pd
import numpy as np

def check_nulls(X): # Checks a dataframe for nulls and reports them in a nice "pretty" format.
    columns = X.columns
    nulls = 0
    for i in range(len(columns)):
        if X.isnull().sum()[i] == 0:
            continue # This way you don't print out every column, just the ones that contain nulls.
        elif X.isnull().sum()[i] == 1:
            print(f'Column "{columns[i]}" contains 1 missing value.') # Separating this out for the singular 'value.'
            nulls += 1
        else:
            print(f'Column "{columns[i]}" contains {X.isnull().sum()[i]} missing values.')
            nulls += X.isnull().sum()[i] # Keeps track of total nulls.
    if nulls == 0:
        print('This dataset contains no missing values!')
    elif nulls == 1:
        print('This dataset contains 1 missing value.')
    else:
        print(f'This dataset contains {nulls} missing values.')

def conf_mat(y_true, y_pred, labels='auto'): # Reports a confusion matrix, with labels for easier interpretation.
    if type(y_true) != np.ndarray:
        print('ERROR: Inputs must be NumPy arrays.')
    elif type(y_pred) != np.ndarray:
        print('ERROR: Inputs must be NumPy arrays.')
    elif len(y_true) != len(y_pred):
        print('ERROR: The arrays must have the same length.')
    else:
        if labels=='auto':
            labels_0 = []
            for i in range(len(np.unique(y_true))):
                labels_0.append(np.unique(y_true)[i])
            for j in range(len(np.unique(y_pred))):
                labels_0.append(np.unique(y_pred)[j])
            labels = list(set(labels_0)) # Putting it into a set first so I only have each unique label once.
        rows = []
        for k in range(len(labels)): # For each label,
            row = [0] * len(labels) # create a row with zeroes, with length equal to the number of unique labels.
            for m in range(len(y_pred)): # For each value in y_pred,
                if y_pred[m] == labels[k]: # if that value is equal to the label under consideration,
                    for n in range(len(labels)):
                        if labels[n] == y_true[m]: # then take the label that value is equal to,
                            row[n] += 1  # and add 1 to the corresponding location in the row.
            rows.append(row)
        row_labels = [f'Predicted {label}' for label in labels]
        column_labels = [f'Actual {label}' for label in labels]
        return pd.DataFrame(data=rows, index=row_labels, columns=column_labels)

def train_val_test_split(X, sizes='auto', random_state=np.random.RandomState(), shuffle=True, stratify=None): # Train/validate/test split
    if sizes == 'auto':
        train_size = 0.7
        val_size = 0.2
        test_size = 0.1
    elif type(sizes) != list:
        print("ERROR: 'sizes' must either be a list of 3 numbers or 'auto.'")
    elif len(sizes) != 3:
        print("ERROR: 'sizes' must either be a list of 3 numbers or 'auto.'")
    elif sum(sizes) != 1:
        print("ERROR: 'sizes' must add up to exactly 1.")
    else:
        train_size = sizes[0]
        val_size = sizes[1]
        test_size = sizes[2]
    
    if shuffle==True:
        if stratify==None:
            Y = X.sample(frac=1, random_state=random_state).reset_index() # Resetting the index so the .loc command works properly.
            test_rows = np.round(len(Y) * test_size)
            if test_rows == 0:
                test_rows == 1 # Setting test_rows and val_rows first to make sure neither equals 0.
            val_rows = np.round(len(Y) * val_size)
            if val_rows == 0:
                val_rows == 1
            test = Y.loc[0:test_rows].set_index('index') # Putting back the original index. User then has the choice to use reset_index themselves.
            del test.index.name # There is no need to name the index "index."
            val = Y.loc[test_rows+1:test_rows+val_rows].set_index('index') # Starting where the test set ended, ending when you reach val_rows total rows.
            del val.index.name
            train = Y.loc[test_rows+val_rows+1:len(Y)].set_index('index') # Starting where the val set ended.
            del train.index.name
        else:
            counts = X[stratify].value_counts()
            train = pd.DataFrame()
            val = pd.DataFrame()
            test = pd.DataFrame()
            for value in range(len(counts)):
                temp = X[X[stratify] == value] # Isolate rows where the stratification column equals a certain value.
                temp2 = temp.sample(frac=1, random_state=random_state).reset_index()
                test_rows = np.round(len(temp2) * test_size)
                if test_rows == 0:
                    test_rows == 1
                val_rows = np.round(len(temp2) * val_size)
                if val_rows == 0:
                    val_rows == 1
                temp_to_test = temp2.loc[0:test_rows].set_index('index')
                del temp_to_test.index.name
                temp_to_val = temp2.loc[test_rows+1:test_rows+val_rows].set_index('index')
                del temp_to_val.index.name
                temp_to_train = temp2.loc[test_rows+val_rows+1:len(temp2)].set_index('index')
                del temp_to_train.index.name
                train = pd.concat([train, temp_to_train]) # Concatenate on each value so we end up with the entire dataset after the loop finishes.
                val = pd.concat([val, temp_to_val])
                test = pd.concat([test, temp_to_test])
    else:
        if stratify != None:
            print("ERROR: If shuffle = 'False,' then stratify must = 'None.'")
        else:
            test_rows = np.round(len(X) * test_size)
            if test_rows == 0:
                test_rows == 1
            val_rows = np.round(len(X) * val_size)
            if val_rows == 0:
                val_rows == 1
            train_rows = len(X) - (test_rows + val_rows)
            train = X.loc[0:train_rows] # Doing train first this time because that's more intuitive when there's no shuffling.
            val = X.loc[train_rows+1, train_rows+val_rows]
            test = X.loc[train_rows+val_rows+1, len(X)]
    
    return train, val, test

def add_list_to_df(list_, df, name='list'): # Single function to add list to dataframe as a new column. (Using 'list_' to avoid overwriting 'list.')
    if type(list_) != list:
        print("ERROR: First parameter must be a list.")
    elif type(df) != pd.core.frame.DataFrame:
        print("ERROR: Second parameter must be a Pandas dataframe.")
    else:
        column = pd.DataFrame(list_, columns=[name])
        return pd.concat([df, column], axis=1)

def split_date(dataframe, column, errors='raise', dayfirst=False, yearfirst=False, utc=None, format=None,
               exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=True, drop=False):
               # Function to split dates into multiple columns.
    if type(dataframe) != pd.core.frame.DataFrame:
        print("ERROR: First parameter must be a Pandas dataframe.")
    else:
        X = dataframe.copy()
        X['new'] = pd.to_datetime(X[column], errors=errors, dayfirst=dayfirst, yearfirst=yearfirst, utc=utc,
                                   format=format, exact=exact, unit=unit,
                                   infer_datetime_format=infer_datetime_format,
                                   origin=origin, cache=cache)
        # Creating a new column so the user has the option not to convert the original column to datetime.
        X[f'{column}_year'] = X['new'].dt.year
        X[f'{column}_month'] = X['new'].dt.month
        X[f'{column}_day'] = X['new'].dt.day
        X = X.drop('new', axis=1)
        if drop==True:
            X = X.drop(column, axis=1)
        return X