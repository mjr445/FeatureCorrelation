import pandas
import numpy as np
import matplotlib.pyplot as plt

df = pandas.read_excel('Matthew_Data.xlsx')  # Import the initial data sheet from excel
R_raw = df  # Store the initial data sheet as a separate variable
corr_mat = R_raw.corr().abs()  # Makes a correlation matrix given the data sheet, converts to absolute values

# Plot the first correlation matrix
plt.figure(1)
plt.matshow(corr_mat, fignum=1)
plt.title("Correlation Matrix Before Feature Selection Using PCC")
plt.xlabel("Features")
plt.ylabel("Features")

AvgCorr = corr_mat.mean()  # Calculates the average
# Set the upper triangle and diagonal of correlation matrix to NaN, negating duplicate pairs
corr_mat.values[np.triu_indices_from(corr_mat, 0)] = np.nan
# Set any ppc value under or equal to 0.9 to Nan, leaving only pairs correlated above 0.9
corr_mat = corr_mat.where(corr_mat > 0.9)
# Turns python data frame into a series, stacks data and removes all NaN values
s = corr_mat.stack()
so = s.sort_values(kind="quicksort")  # Sorts the values in ascending order
drop_columns = set()  # Creates a set that will hold the names of the features to be dropped

for x, val in so.iteritems():  # Iterates through correlation pairs > 0.9
    # Features average correlations are compared; feature with higher average correlation is dropped
    if AvgCorr[x[0]] > AvgCorr[x[1]]:
        drop_columns.add(x[0])
    else:
        drop_columns.add(x[1])

print(drop_columns)  # Prints the name of the features that have been dropped
df = df.drop(drop_columns, axis=1)  # Drops the columns from the original data sheet
with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())  # Prints head of the data sheet so that it can be checked for accuracy

# Plots the new correlation matrix after feature selection has been performed
plt.figure(2)
plt.matshow(df.corr().abs(), fignum=2)
plt.title("Correlation Matrix After Feature Selection Using PCC")
plt.xlabel("Features")
plt.ylabel("Features")

plt.show()
