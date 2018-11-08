import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import itertools
from matplotlib.patches import Ellipse
import matplotlib as mpl
from sklearn.mixture import GaussianMixture as GMM

# df is the data file that is being manipulated

#  Following function is from the scikit learn site on GaussianMixture model
def plot_results(X, Y_, means, covariances, title):  # add index between covariance and title if you want subplots.
    #splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        #ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        #splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.axis('on')

df = pd.read_excel('Matthew_Data.xlsx')  # Import the initial data sheet from excel
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


# Plots the new correlation matrix after feature selection has been performed
plt.figure(2)
plt.matshow(df.corr().abs(), fignum=2)
plt.title("Correlation Matrix After Feature Selection Using PCC")
plt.xlabel("Features")
plt.ylabel("Features")

plt.show()



#  The following code fits a GaussianMixture model on the cleaned data and graphs it.
GaussianMix=GMM(n_components=2,covariance_type='full',max_iter=800,init_params='random').fit(df)
color_iter = itertools.cycle(['navy', 'c'])
plot_results(np.array(df), GaussianMix.predict(df),GaussianMix.means_,GaussianMix.covariances_,"Gaussian Mixture")
plt.show()
