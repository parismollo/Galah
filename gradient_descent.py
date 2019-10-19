# Starting by the idea of the Gradient Descent
from typing import List, Iterator, TypeVar
import random
from linear_algebra import scalar_multiply, vector_mean, add

Vector = List[float]

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


T = TypeVar("T")

def minibatches(dataset: List[T], batch_size: int, shuffle: bool = True) -> Iterator[List[T]]:
    batch_starts = [start for start in range(0, len(dataset),batch_size)]
    if shuffle: random.shuffle(batch_starts)

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(1000):
    for batch in minibatches(inputs, batch_size=20):
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

slope, intercept = theta

assert 9.99 < slope < 10.11, "slope should be about 10"
assert 6.9 < intercept < 7.1,"intercept should be about 7"


theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
for epoch in range(100):
    for x, y in inputs:
        grad = linear_gradient(x, y, theta)
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

slope, intercept = theta

assert 9.99 < slope < 10.11, "slope should be about 10"
assert 6.9 < intercept < 7.1,"intercept should be about 7"
