from IPython.display import display
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Lock, Pool
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import seaborn as sns
import os

data_dir = "./data/"
os.makedirs(data_dir) if not os.path.exists(data_dir) else None
plot_dir = "./plots/"
os.makedirs(plot_dir) if not os.path.exists(plot_dir) else None

processes = cpu_count() - 1

# display data frames, side by side in jupyter
def display_side_by_side(*args,titles=cycle([''])):
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2 style="text-align: center;">{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)

def to_pickle(data, path):
    with open(path, 'wb') as file:
        return pickle.dump(data, file)

def read_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def to_csv(df, path):

    # if index is not default, reset
    if not all(df.index == range(len(df))):
        df = df.reset_index()
    
    # Prepend dtypes to the top of df (from https://stackoverflow.com/a/43408736/7607701)
    df.loc[-1] = df.dtypes
    df.index = df.index + 1
    df.sort_index(inplace=True)

    # Then save it to a csv
    df.to_csv(path, index=False)

def read_csv(path):
    # Read types first line of csv
    dtypes = pd.read_csv(path, nrows=1).iloc[0].to_dict()
    # Read the rest of the lines with the types from above
    return pd.read_csv(path, dtype=dtypes, skiprows=[1])


def df_info(df: pd.DataFrame):

    """
    print info about about a dataframe
    
    parameters:
        df, pd.DataFrame : data frame to analyze
    
    returns:
        prints out the info
    """

     # shape
    print('\tshape')
    print('-'*(8+len('shape')+8))
    print('rows:', '{:,}'.format(df.shape[0]))
    print('cols:', '{:,}'.format(df.shape[1]))
    print()

    # head
    print('\thead')
    print('-'*(8+len('head')+8))
    display(df.head())
    print()

    # info
    print('\tinfo')
    print('-'*(8+len('info')+8))

    info = pd.DataFrame(columns=['column name', 'dtype', 'count', '# null', 'null %', '# unique', 'top', 'top freq', 'top %'])
    for column in df.columns:
    
        col = df[column]
        ln = len(col)
        
        # skip nulls
        i = 0
        while i < ln-1 and pd.isnull(col.iloc[i]):
            i += 1

        # get dtype
        dtype = type(col.iloc[i])
    
        # set str type
        strtype = 'single'
        if dtype == str:
            if '\n' in col.iloc[i]:
                strtype = 'multi'
    
        # loop through column (skipping nulls)
        for j in range(i+1,ln):
            c = col.iloc[j]
            if pd.notnull(c):
                # check dtype
                if dtype != type(c):
                    dtype = 'mixed'
    
                # check strtype
                if dtype == type(''):
                    if '\n' in c:
                        strtype = 'multi'
    
        # add strtype
        if dtype != 'mixed':
            dtype = dtype.__name__
        if dtype == 'str':
            dtype += ' - ' + strtype if strtype == 'multi' else ''
            
        nulls = sum(pd.isnull(col))
        values = col.value_counts()
        values = '-' if values.empty else values
        uniques = col.unique()
        
        info.loc[len(info)] = {
            'column name': column,
            'dtype': dtype,
            'count': ln - nulls,
            '# null': nulls,
            'null %': round(nulls/ln*100,2),
            '# unique': sum(pd.notnull(uniques)),
            'top': values if type(values) == str else values.index[0], 
            'top freq': values if type(values) == str else values.iloc[0],
            'top %': values if type(values) == str else round(values.iloc[0]/ln*100,2)
        }
        
    info = info.set_index('column name')

    for i in range(0,len(df.columns),25):
        display(info.iloc[i:i+25])
    print()
    
    # # info
    # print(df.info(show_counts=True))

    # describe
    print('\tdescribe')
    print('-'*(8+len('describe')+8))
    
    desc = df[info[info['null %'] != 100].index].describe().transpose()
    for i in range(0,len(desc),25):
        display(desc.iloc[i:i+25])
        
    print()

def getCategoricals(df: pd.DataFrame):

    """
    identifies columns of df as categorical variables
    
    parameters:
        df, pd.DataFrame : data frame to analyze
        
    returns:
        array of column names identified as categorical
    """
    
    cols = []
    for column in df.columns:
        if df[column].dtype == object:
            cols.append(column)
            
    return cols

def listToDummy(s: pd.Series, prefix='', suffix=''):
    df = pd.get_dummies(s.explode()).groupby(level=0).sum()
    df.columns = [prefix + col + suffix for col in df.columns]
    return df

def listStrToDummy(s: pd.Series, sep=',', prefix='', suffix=''):
    df =  pd.get_dummies(s.apply(lambda x: x.replace("'",'')[1:-1].split(sep)).explode()).groupby(level=0).sum()
    df.columns = [prefix + col + suffix for col in df.columns]
    return df

def commaColToDummy(df: pd.DataFrame, column: str, sep=',', prefix='', suffix=''):

    """
    turns a column with comma separated values into dummy variables
    - new dummy column names are the unique values with prefix and/or suffix added
    
    parameters:
        df, pd.DataFrame : data frame to alter
        column, str : column to turn into dummies
        sep, str : the separator of the column values
        prefix, str : prefix of the new dummy column names
        suffix, str : suffix of the new dummy column names
        
    returns:
        the original data frame concatenated with the dummy columns
    """
   
    # column of interest
    series = df[column]
   
    # get unique values in column of interest and create a new data frame
    new_columns = set([prefix + new_col + suffix for row in series.unique() for new_col in row.split(sep)])
    new_df = pd.DataFrame(columns=new_columns)
   
    # for each row
    for row in series:
        # make a new row
        new_row = [0]*len(new_columns)
       
        # for each unique value in that row
        for req in row.split(sep):
            for i,col in enumerate(new_df.columns):
                # find unique value's column
                if req == col[len(prefix):len(col)-len(suffix)]:
                    # set column to 1
                    new_row[i] = 1
                    break
               
        # append new row
        new_df.loc[len(new_df)] = new_row
   
    # add new cols to original data frame
    return pd.concat([df,new_df], axis=1)

"""
turns a Series of json strings into dummy variables for each key in the jsons

parameters:
    jsonColumn, pd.Series : a pandas Series of json strings

returns:
    a pandas DataFrame of the dummy variables
"""
def json2dummies(jsonColumn: pd.Series):
    # data frame with business_ids
    dummies = pd.DataFrame(businesses.business_id)
    
    # keys to be used as column names
    key_columns = []
    
    # loop through jsons
    for i,json in enumerate(jsonColumn):
        
        # validate - not null, empty, none
        if pd.notnull(json) and json != "{}" and json != "None":
            
            # get key-value pairs
            keypairs = []
            if type(json) == str:
                keypairs = [pair.split(": ") for pair in json[1:-1].split(", ")]
            elif type(json) == dict:
                keypairs = json.items()
            
            # loop through key pairs
            for key,val in keypairs:
                
                # clean the key
                if key[0] == 'u':
                    key = key[1:]
                if key[0] == "'" and key[-1] == "'":
                    key = key[1:-1]
                
                # make the new column name
                column_name = key
                if jsonColumn.name != "attributes":
                    column_name = jsonColumn.name + '_' + column_name
                
                # if new key, add new column
                if column_name not in key_columns:
                    key_columns += [column_name]
                    
                    # add new column
                    dummies[column_name] = [np.nan] * len(dummies)
            
                # add value to key column
                dummies.loc[i,column_name] = val
                
    return dummies
