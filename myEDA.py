from IPython.display import display
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Lock, Pool
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import os

data_dir = "./data/"
os.makedirs(data_dir) if not os.path.exists(data_dir) else None
plot_dir = "./plots/"
os.makedirs(plot_dir) if not os.path.exists(plot_dir) else None

processes = cpu_count() - 1


"""
print info about about a dataframe

parameters:
    df, pd.DataFrame : data frame to analyze

returns:
    prints out the info
"""
def df_info(df: pd.DataFrame):
    print(df.info(show_counts=True))
    display(df.describe().transpose())
    display(df.head())
    print(df.shape)

"""
identifies columns of df as categorical variables

parameters:
    df, pd.DataFrame : data frame to analyze
    
returns:
    array of column names identified as categorical
"""
def getCategoricals(df: pd.DataFrame):
    cols = []
    for column in df.columns:
        if df[column].dtype == object:
            cols.append(column)
            
    return cols


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
def commaColToDummy(df: pd.DataFrame, column: str, sep=",", prefix="", suffix=""):
   
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

"""
plots numerical columns in blue as histograms, plots categorical columns in orange as bar charts, displays and saves the figure to plot directory

parameters:
    df, pd.DataFrame : data frame to analyze
    cat_cols, [str] : array of column names identified as categorical
    
returns:
    displays and saves the image to plots directory as histogram.png
"""
def histograms(df, cat_cols=[]):
    cols = 5
    rows = int(np.ceil(float(df.shape[1]) / cols))

    fig = plt.figure(figsize=(3.75*cols,3.75*rows))

    for i, column in enumerate(df.columns):

        # set up subplot
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)

        # categorical vs continuous
        if column in cat_cols:
            cats = df[column].value_counts().sort_index()
            plt.bar(cats.index, cats, width=0.1, color="orange")
            plt.plot(cats)
        else:
            df[column].hist(axes=ax, bins=50)

        plt.xticks(rotation=45)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.savefig(plot_dir + "histograms.png")


def histograms2(df, cat_cols=[]):
    cols = 5
    rows = int(np.ceil(float(df.shape[1]) / cols))
    columns = df.columns

    # set up subplots
    fig, axs = plt.subplots(figsize=(3.75*cols,3.75*rows), nrows=rows, ncols=cols)

    # delete subplots
    d = cols - len(columns)%cols
    for i in range(1,d+1):
        fig.delaxes(axs[-1,-i])
    axs = np.ravel(axs)[:-d]

    with Pool(processes) as pool:
        pool.starmap(hist_process, [(i,df[columns[i]],ax,cat_cols) for i,ax in enumerate(axs)])

    for i,ax in enumerate(axs):
        ax.set_title(columns[i])
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(plot_dir + "histograms.png")

"""

"""
def hist_process(i,column,ax,cat_cols):    

    # set axes
    plt.sca(ax)
    
    # categorical column
    if column in cat_cols:
        cats = column.value_counts().sort_index()
        plt.bar(cats.index, cats, width=0.1, color="orange")
        plt.plot(cats)
    # continuous column
    else:
        plt.hist(column, bins=50)
    
    
"""
plots each column as a boxplot, displays and saves the figure to plot directory

parameters:
    df, pd.DataFrame : data frame to analyze
    
returns:
    displays and saves the image to plots directory as histogram.png
""" 
def boxplots(df):
    cols = 5
    rows = int(np.ceil(float(df.shape[1]) / cols))

    fig = plt.figure(figsize=(3.75*cols,3.75*rows))

    for i, column in enumerate(df.columns):

        ax = fig.add_subplot(rows, cols, i + 1)

        df[column].plot.box(ax=ax, sym='k_')

        plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.savefig("./plots/boxplots.png")

    
"""
plots scatter plots between all columns in the data frame, displays and saves the figure to plot directory

parameters:
    df, pd.DataFrame : data frame to analyze
    color, str : column to use for color
    cmap, str : colormap
    
returns:
    displays and saves the image to plots directory as scatterplots.png
"""
def scatterplots(df: pd.DataFrame, color: str, cmap="plasma"):
    plots = df.shape[1]

    fig = plt.figure(figsize=(3.75*plots,3.75*plots))
    fig.tight_layout()

    for i, columnY in enumerate(df.columns):
        for j, columnX in enumerate(df.columns):

            ax = fig.add_subplot(plots, plots, i*plots + j + 1)

            if columnX == columnY:
                df[columnX].hist(color="red", axes=ax, bins=50)
            else:
                df.plot.scatter(x=columnX, y=columnY, ax=ax, s=1,
                                c=df[color], cmap=cmap)

            # axis labels
            ax.set_xlabel("")
            ax.set_ylabel("")
            if j == 0:
                ax.set_ylabel(columnY)
            if j % 2 == 1:
                ax.set_xlabel(columnX)
            if i == df.shape[1] - 1:
                ax.set_xlabel(columnX)

            plt.subplots_adjust(hspace=0.3, wspace=0.25)

    plt.savefig("./plots/scatterplots.png")