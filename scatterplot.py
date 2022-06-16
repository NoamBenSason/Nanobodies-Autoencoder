import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_mse_scatterplot(sequence_vec, structure_vec):
    """
    Creates a scatter plot of samples where X axis is the MSE structure and the
     Y axis is the MSE sequence
    :param sequence_vec: A vector of size 1 X 50 of the MSE of the sequence for
     each sample.
    :param structure_vec: A vector of size 1 X 50 of the MSE of the structure
    for each sample.
    :return:
    """

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    N = 5
    x = np.random.rand(N)
    y = np.random.rand(N)
    colors = np.random.rand(N)
    area = (30 * np.random.rand(N)) ** 2  # 0 to 15 point radii

    plt.scatter(structure_vec, sequence_vec, alpha=0.5)
    plt.title("MSE of the structure vs MSE of the sequence")
    plt.legend()
    plt.xlabel("MSE structure")
    plt.ylabel("MSE sequence")
    plt.show()

if __name__ == '__main__':
    create_mse_scatterplot([0.3, 0.6, 0.9, 0.32, 0.72], [0.46, 0.81, 0.17, 0.36, 0.78])
