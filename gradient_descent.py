# Starting by the idea of the Gradient Descent
from typing import List
import random
Vector = List[float]

def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "Vector must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [1, 2, 3]) == [2, 4, 6]


def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

def vector_sum(vectors: List[Vector]) -> Vector:
    assert vectors, "no vectors provided"
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes"
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

inputs = [(x, 10 * x + 7) for x in range(-50, 50)]

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept
    error = (predicted - y)
    squared_error = error ** 2
    grad = [2 * error * x, 2 * error]
    return grad

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

learning_rate = 0.001

for epoch in range(5000):
    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    theta = gradient_step(theta, grad, -learning_rate)

    print(epoch, theta)

slope, intercept = theta
assert 9.99 < slope < 10.11, "slope should be about 10"
assert 6.9 < intercept < 7.1,"intercept should be about 7"
