# ----- imports -----
import numpy as np
import pandas as pd
import logomaker as lm
import matplotlib.pyplot as plt
plt.ion()




if __name__ == '__main__':
    # load probability matrix
    prob_df = pd.read_csv("prob_csv_1dlf_model_2.csv")
    prob_df = prob_df.drop(["pos"], axis=1)
    prob_df.head()


    # create and style logo
    colors_dict = {"A":"#FEA89A", "C":"#FCE59C", "D":"#C0FE9A", "E":"#9CFCD7",
               "F":"#9DD9FB", "G":"#B49FF9", "H":"#FA9CFC", "I":"#FE30CD",
               "K":"#9231FD", "L":"#3680F8", "M":"#19F7A8", "N":"#9EFA16",
               "P":"#FB8815", "Q":"#AD5803", "R":"#9AA907", "S":"#06AA2D",
               "T":"#0587AB", "W":"#1202AE", "Y":"#A808AC", "V":"#5C045E",
               "X":"#080008", "-":"#080008"}
    logo = lm.Logo(df=prob_df,
                   font_name='Hobo Std',
                   fade_below=0,
                   shade_below=0,
                   figsize=(200,20), color_scheme=colors_dict)

    # set axes labels
    logo.ax.set_xlabel('Position',fontsize=14)
    logo.ax.set_ylabel("Probability", labelpad=-1,fontsize=14)
