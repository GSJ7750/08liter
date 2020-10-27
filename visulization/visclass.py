#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline,BSpline

class vis():
    def __init__(self):
        None
    def make_vis(self, X, Y, alpha = 0.3, n = 100, fig_size=(10,5), xlabel='', ylabel='', title='', scatter=None):
        xnew = [np.linspace(x.min(), x.max(), n) for x in X]
        spl = [make_interp_spline(x, y, k=2) for x, y in zip(X,Y)]
        y_smooth = [spl_e(x) for x, spl_e in zip(xnew,spl)]

        
        plt.figure(figsize=fig_size)
        plt.title(title,fontsize=fig_size[0])
        plt.xlabel(xlabel,fontsize=fig_size[0])
        plt.ylabel(ylabel, rotation=0,fontsize=fig_size[0])
        [plt.fill_between(x,0,y,alpha=alpha) for x, y in zip(xnew, y_smooth)]
        [plt.plot(x, y, alpha=alpha+0.1) for x, y in zip(xnew, y_smooth)]
        if scatter == True:
            [plt.scatter(x, y) for x, y in zip(X,Y)]
        else:
            pass
        plt.margins(0,0)
        
    def make_subplots(self, X, Y, alpha = 0.3, n = 100, fig_size=(10,5), xlabel='', ylabel='', columns=None, scatter=None):
        xnew = [np.linspace(x.min(), x.max(), n) for x in X]
        spl = [make_interp_spline(x, y, k=2) for x, y in zip(X,Y)]
        y_smooth = [spl_e(x) for x, spl_e in zip(xnew,spl)]
        
        
        for i in range(len(xnew)):
            plt.figure(figsize=fig_size)
            plt.subplot(len(xnew),1,i+1)
            plt.xlabel(xlabel,fontsize=fig_size[1])
            plt.ylabel(ylabel, rotation=0,fontsize=fig_size[1])
            if columns:
                plt.title(columns[i],fontsize=fig_size[1])
            plt.fill_between(xnew[i],0,y_smooth[i], alpha=alpha)
            plt.plot(xnew[i],y_smooth[i], alpha=alpha+0.1)
            if scatter == True:
                plt.scatter(X[i], Y[i])
            else:
                pass
        
        
        
        
        
if __name__ == '__main__':
    None

