# Gradient Descent

The purpose of this repository is to practice the use of the gradient descent technique. Exercise from the Book "Data Science from Scratch".


## What  have I learned from Gradient Descents?

Frequently when we will need to find a model that will do accurate forecasts of a specific data set, we will need to minimize its errors and the technique to do so is called Gradient Descent.

We can perceive this problem as if we were trying to find the inputs of a specific function that will optimize its output

There are many ways of computing the error of a function, we will use the sum of squared residuals. The first step is to compute the partial derivatives of the loss function for each parameter, e.g. slope and the intercept, in a more machine learning language, we are computing the gradients of the loss function. We will then select random values for the parameters and plug them into the gradients.

The gradient with the parameters (theta) will give us the slope of the loss function, we will then compute the “step size” which is nothing more than the slope of the gradient times the learning rate. The step size will give us the value of the next value for (theta) by doing :
new (theta) = old (theta) - step size. Then we will repeat this process until we find a slope that gets close to 0, therefore, a step size smaller than 0.001 and this should give us the value of (theta) that will fit our line (if linear gradient) the best in our data set with smallest SSR (sum of squared residuals)

* Stochastic gradient descent (todo)
* Mini-batches (todo)
