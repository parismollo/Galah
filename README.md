# Gradient Descent

The purpose of this repository is my self practice of the Gradient Descent technique. Exercise from the Book "Data Science from Scratch"

I am a student that is learning, let me know if you find any errors,the code is inspired from examples and exercises found in books.



## What  have I learned from Gradient Descent?

Frequently when we will need to find a model that will do accurate forecasts of a specific data set, we will need to minimize its errors and the technique to do so is called Gradient Descent.

We can perceive this problem as if we were trying to find the inputs of a specific function that will optimize its output

There are many ways of computing the error of a function, we will use the sum of squared residuals. The first step is to compute the partial derivatives of the loss function for each parameter, e.g. slope and the intercept, in a more machine learning language, we are computing the gradients of the loss function. We will then select random values for the parameters and plug them into the gradients.

The gradient with the parameters (theta) will give us the slope of the loss function, we will then compute the “step size” which is nothing more than the slope of the gradient times the learning rate. The step size will give us the value of the next value for (theta) by doing :
new (theta) = old (theta) - step size. Then we will repeat this process until we find a slope that gets close to 0, therefore, a step size smaller than 0.001 and this should give us the value of (theta) that will fit our line (if linear gradient) the best in our data set with smallest SSR (sum of squared residuals)

* Stochastic gradient descent

If we have a big data set, compute the values of each data point for all the different values of (theta) can be extremely expensive to compute, Stochastic gradient Descent takes a single random value of the data set for each gradient computation.
* Mini-batches

Mini-batches has the best of both worlds, will create random "batches" of the data set for each computation in order to learn more from the data in comparison to the Stochastic technique, however it will take a limited size batch in order to avoid large expensive computation in comparison to the normal Gradient Descent

# Todo
- [ ] Difference quotient
- [ ] Partial derivatives quotient
- [ ] add mountain situation 
## Run the code

1. Clone the repository
2. Make sure you are using python 3.6
3. Run the main file
4. To fit with the linear model choose one of the 3 options (1) Gradient Descent (2) Mini-batches (3) Stochastic



## Resources that I used to learn about this fun topic:
- Book: Data Science from Scratch, Joel Grus
- Video: StatQuest with Josh Starmer
