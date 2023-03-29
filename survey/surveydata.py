import pandas as pd
import matplotlib.pyplot as plt

def column_mean_plot(df, column):
    # Plot the 'realistic' column of the dataframe
    df.plot(x='img nr', y=column, kind='bar')

    # Set the axis labels and title
    plt.xlabel('Image Number')
    plt.ylabel(f'{column} Score')
    plt.ylim([1,5])
    plt.title(f'{column} Scores by Image')

    # Show the plot
    plt.show()



# Load the CSV file into a dataframe
responses = pd.read_csv('responses.csv')




#  999 -  real image
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
        'img nr': [46, 269, 28, 30, 154, 999, 224, 155, 17, 152, 199, 223, 999, 999, 201],
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
    df.loc[df['question'] == i, 'creative'] = realistic
    likeness = responses[str(i+0.4)].mean()
    df.loc[df['question'] == i, 'likeness'] = likeness
    connection = responses[str(i+0.5)].mean()
    df.loc[df['question'] == i, 'connection'] = connection


df = df.sort_values(by='img nr')
column_mean_plot(df, "realistic")
column_mean_plot(df, "creative")
column_mean_plot(df, "likeness")
column_mean_plot(df, "connection")
# plt.show()





