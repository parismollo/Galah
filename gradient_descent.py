# Starting by the idea of the Gradient Descent
import matplotlib.pyplot as plt
from typing import List

Vector = List[float]
def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), "Vectors must be the same length"
    return sum(v_i * w_i for v_i, w_i in zip(v,w))

assert dot([1, 2, 3], [2, 2, 2]) == 12, "Something went wrong"

from typing import Callable

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
