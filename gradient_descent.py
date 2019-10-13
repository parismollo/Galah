# Starting by the idea of the Gradient Descent
import matplotlib.pyplot as plt
from typing import List, Callable
import random
import math


Vector = List[float]


def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "Vector must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [1, 2, 3]) == [2, 4, 6]

def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), "Vectors must be the same length"
    return sum(v_i * w_i for v_i, w_i in zip(v,w))

assert dot([1, 2, 3], [2, 2, 2]) == 12, "Something went wrong"


def sum_of_squares(v: Vector) -> float:
    return dot(v, v)


def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]


def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "Both should have the same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]

assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]


def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v))

assert magnitude([3, 4]) == 5

def squared_distance(v: Vector, w: Vector) -> float:
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    return math.sqrt(squared_distance(v, w))

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h

def square(x: float) -> float:
    return x * x

def derivative_sqrt_function(x: float) -> float:
    return 2 * x


xs = range(-10, 11)
actuals = [derivative_sqrt_function(x) for x in xs]
estimates = [difference_quotient(square, x, h=0.001) for x in xs]

plt.title("Actual Derivatives vs. Estimates")
plt.plot(xs, actuals, 'rx', label='Actual')       # red  x
plt.plot(xs, estimates, 'b+', label='Estimate')   # blue +
plt.legend(loc=9)
# plt.show()

def partial_difference_quotient(f: Callable[[Vector], float], v: Vector, i: int, h: float)-> float:
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v) / h)

def estimate_gradient(f: Callable[[Vector], float], v: Vector, h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)


def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

v = [random.uniform(-10, 10) for i in range(3)]

for epoch in range(1000):
    grad = sum_of_squares_gradient(v)
    v = gradient_step(v, grad, -0.01)
    print(epoch, v)

assert distance(v, [0, 0, 0]) < 0.001 
