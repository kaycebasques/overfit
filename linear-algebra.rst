.. _linear-algebra:

==============
Linear algebra
==============

.. _Linear Algebra for Machine Learning and Data Science: https://www.coursera.org/learn/machine-learning-linear-algebra?specialization=mathematics-for-machine-learning-and-data-science

These are my notes from `Linear Algebra for Machine Learning and Data
Science`_.

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
input and output is linear. I.e. when there's a single input and a single
output, you can literally plot the relationship as a straight line.
Here's the wind farm example plotted as a line:

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

Suppose you want to add a second input variable, "temperature".
Your table of data becomes this:

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

Where :math:`n` is the number of input variables. In ML we call each input
variable a "feature". The output variable that you want to predict is
called a "label".

Now that there are 3 variables, the relationship can no longer be plotted as
a line on a 2-dimensional grid. It has to be graphed as a plane in
three-dimensional space. Every variable you add requires a new dimension.
And notice also that the relationship went from a one-dimensional line to a
two-dimensional plane.

(Add a screenshot here.)

(Maybe it's correct to say that for an equation with :math:`m` dimensions
the relationship will be in the :math:`m-1` dimension?)

In real datasets you have many records. Each record could be represented
with its own equation:

.. math::

   y^{(1)} = w_1x_1^{(1)} + w_2x_2^{(1)} + \ldots + w_nx_n^{(1)} + b

   y^{(2)} = w_1x_1^{(2)} + w_2x_2^{(2)} + \ldots + w_nx_n^{(2)} + b

   y^{(3)} = w_1x_1^{(3)} + w_2x_2^{(3)} + \ldots + w_nx_n^{(3)} + b

   \vdots

   y^{(m)} = w_1x_1^{(m)} + w_2x_2^{(m)} + \ldots + w_nx_n^{(m)} + b

The superscript just denotes a record. It's not an exponent.

This collection of records is called a "system of linear equations".
Solving the system means finding weights and biases that satisfy every
linear equation in the system simultaneously. Or weights and biases
that at least get *close* to solving each equation.
