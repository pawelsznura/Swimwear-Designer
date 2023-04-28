import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def distrubitionPlot(df):

    
    sns.histplot(data=df, x='score', kde=True)
    plt.xlabel('Image score')
    plt.ylabel('occurances' )
    plt.ylim([1,5])
    plt.title('score distribution ')

    plt.show()

# Load the CSV file into a dataframe
df = pd.read_csv('..\img_evaluation.csv')


# Exclude the first x rows
# df = df.iloc[50:]

# Exclude rows with score "0"
# df = df[df['score'] != 0]


# df.sort_values(by='score').plot(x='image', y='score', kind='bar')
df.plot(x='image', y='score', kind='bar')

# distrubitionPlot(df)

# Set the axis labels and title
plt.xlabel('Image Number')
plt.ylabel(' Score')
# plt.ylim([1,5])
plt.title(' Scores by Image')

# Show the plot
plt.show()

print(df)

