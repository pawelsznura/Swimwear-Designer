import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

def column_mean_plot(df, column, sort=False, limiter=True):
    # Plot the selected column of the dataframe
    # limiter is to set the scale for 1-5

    if sort:
        df = df.sort_values(by=column)

    # Set the color of the bars to orange
    ax = df.plot(x='img nr', y=column, kind='bar', color='#FF7F0E', width=0.6)

    # Set the axis labels and title
    ax.set_xlabel('Image', fontsize=12)
    ax.set_ylabel(f'{column} Score', fontsize=12)
    if limiter:
        ax.set_ylim([1, 5])
    ax.set_title(f'Mean {column} Scores by Image', fontsize=14)

    # Add grid lines to the plot
    plt.grid(axis='y', alpha=0.5)

    # Show the plot
    plt.show()

def getBestImage(df):
    # sums the mean 'realistic, creative, likeness, connection' values
    # the higher the number the better the image in all categories
    
    bestDf = df.copy()
    bestDf['sum'] = bestDf['realistic'] + bestDf['creative'] + bestDf['likeness'] + bestDf['connection']

    bestDf = bestDf.sort_values(by='sum')
    column_mean_plot(bestDf, 'sum', sort=True, limiter=False)

def getGeneratedImages(df):
    # the nongenerated images have the imgnr 991, 992, 993

    # Define the non-generated image numbers
    non_gen = [991, 992, 993]

    # Filter the DataFrame to include only rows with non-generated image numbers
    newDf = df[~df['img nr'].isin(non_gen)]
    
    # Return the filtered DataFrame
    return newDf

def getRealImages(df):
    # the nongenerated images have the imgnr 991, 992, 993

    # Define the non-generated image numbers
    non_gen = [991, 992, 993]

    # Filter the DataFrame 
    newDf = df[df['img nr'].isin(non_gen)]
    
    # Return the filtered DataFrame
    return newDf

def mean_plot(df1, df2=None):
    # Calculate the mean of each column for the first dataframe
    mean_df1 = df1.mean().loc[['realistic', 'creative', 'likeness']]

    # Create a numpy array of the mean values for the first dataframe
    values1 = np.array(mean_df1.values)

    # Set the x-axis labels for the first dataframe
    labels1 = mean_df1.index

    # Create a bar chart with the mean column values for the first dataframe
    plt.bar(labels1, values1, width=0.3, label='Real')

    if df2 is not None:
        # Calculate the mean of each column for the second dataframe
        mean_df2 = df2.mean().loc[['realistic', 'creative', 'likeness', 'connection']]

        # Create a numpy array of the mean values for the second dataframe
        values2 = np.array(mean_df2.values)

        # Set the x-axis labels for the second dataframe
        labels2 = mean_df2.index

        # Create a bar chart with the mean column values for the second dataframe
        plt.bar(labels2, values2, width=0.3, label='Generated', align='edge')

    # Set the plot title and axis labels
    plt.title('Comparison of Mean Ratings', fontsize=14)
    # plt.xlabel('Categories', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.ylim([1, 5])

    # Add a legend to the plot
    plt.legend(fontsize=12)

    # Add grid lines to the plot
    plt.grid(axis='y', alpha=0.5)

    # Show the plot
    plt.show()



# Load the CSV file into a dataframe
responses = pd.read_csv('responses.csv')

# print(responses)


#  999 -  real image
#
# question	img nr
# 1	        46
# 2	        269
# 3	        28
# 4	        30
# 5	        154
# 6	        999
# 7	        224
# 8	        155
# 9	        17
# 10	    152
# 11	    199
# 12	    223
# 13	    999
# 14	    999
# 15	    201

# Create a dictionary with column data
data = {'question': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'img nr': [46, 269, 28, 30, 154, 991, 224, 155, 17, 152, 199, 223, 992, 993, 201],
        # 'aigen':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'realistic':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'creative':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'likeness':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'connection':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # 'comments':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }

df = pd.DataFrame(data)

# Loop through the questions 1 to 15
for i in range(1,16):
    #  Compute the mean value of the 'realistic' column for the current question
    realistic = responses[str(i+0.2)].mean()
    # Assign the mean value to the 'realistic' column of the corresponding row in the dataframe
    df.loc[df['question'] == i, 'realistic'] = realistic
    creative = responses[str(i+0.3)].mean()
    df.loc[df['question'] == i, 'creative'] = creative
    likeness = responses[str(i+0.4)].mean()
    df.loc[df['question'] == i, 'likeness'] = likeness
    connection = responses[str(i+0.5)].mean()
    df.loc[df['question'] == i, 'connection'] = connection


df = df.sort_values(by='img nr')


dfGen = getGeneratedImages(df)
dfReal = getRealImages(df)

# print(dfGen)

# mean_plot(dfReal, dfGen)
# mean_plot(dfGen)

# column_mean_plot(df, "realistic", sort=True)
# column_mean_plot(df, "creative",  sort=True)
# column_mean_plot(df, "likeness",  sort=True)
# column_mean_plot(df, "connection",  sort=True)
# print(df)

column_mean_plot(dfGen, "realistic", sort=False)
# column_mean_plot(dfGen, "creative",  sort=False)
# column_mean_plot(dfGen, "likeness",  sort=False)
# column_mean_plot(dfGen, "connection",  sort=False)



# getBestImage(df)






