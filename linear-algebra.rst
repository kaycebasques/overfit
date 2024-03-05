.. _linear-algebra:

==============
Linear algebra
==============

-----------------
Linear regression
-----------------

Linear regression is a technique for making predictions. It's used in supervised
machine learning. "Supervised" means that you've already collected input and output
data and you want to create a machine learning (ML) model that can predict reasonable
output data when you give it new input data. In other words, you want to find the
relationship between the input and output data. For example, suppose that you run a wind
farm and you've collected the following data:

.. raw:: html

   <table>
     <tr>
       <th>Wind Speed</th>
       <th>Power Generation</th>
     </tr>
     <tr>
       <td>2</td>
       <td>0</td>
     </tr>
     <tr>
       <td>4</td>
       <td>500</td>
     </tr>
     <tr>
       <td>5.25</td>
       <td>2000</td>
     </tr>
     <tr>
       <td>7.5</td>
       <td>2750</td>
     </tr>
     <tr>
       <td>9</td>
       <td>3600</td>
     </tr>
   </table>

On a future date, suppose the wind speed is 6. With linear regression you
have a way to predict the expected power generation when the wind is that
speed.

When you do linear regression you assume that the relationship between the
input and output is linear. When there's one input and one output that means
the relationship can literally be plotted 

You assume that the relationship is linear, i.e. it can literally be plotted
as a line.

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   x = np.array([2, 4,   5.25, 7.5,  9])
   y = np.array([0, 500, 2000, 2750, 3600])
   m, b = np.polyfit(x, y, 1)
   fitted_x = np.linspace(min(x), max(x), 100)
   fitted_y = m * fitted_x + b

   plt.scatter(x, y)
   plt.plot(fitted_x, fitted_y)
   plt.grid(True)
   plt.show()

The line doesn't perfectly fit the data points, but it's pretty close.

In high school algebra this equation is represented like this:

.. math::

   y = mx + b

In machine learning (ML) it's represented like this:

.. math:: 

   y = wx + b

:math:`w` stands for "weight" and :math:`b` stands for "bias".

Suppose you want to add a second input variable, "temperature":

.. raw:: html

   <table>
     <tr>
       <th>Wind Speed</th>
       <th>Temperature</th>
       <th>Power Generation</th>
     </tr>
     <tr>
       <td>2</td>
       <td>80</td>
       <td>0</td>
     </tr>
     <tr>
       <td>4</td>
       <td>70</td>
       <td>500</td>
     </tr>
     <tr>
       <td>5.25</td>
       <td>65</td>
       <td>2000</td>
     </tr>
     <tr>
       <td>7.5</td>
       <td>55</td>
       <td>2750</td>
     </tr>
     <tr>
       <td>9</td>
       <td>40</td>
       <td>3600</td>
     </tr>
   </table>

The equation now looks like this:

.. math::

   y = w_1x_1 + w_2x_2 + b

You can represent the equation generally like this:

.. math::

   y = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b

Where :math:`n` is the number of features.