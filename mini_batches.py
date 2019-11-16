from gradients import linear_gradient, gradient_step, minibatches
from linear_algebra import vector_mean
import random
import time
inputs = [(x, 10 * x + 7) for x in range(-50, 50)]


def run():
    print("computing random values for theta")
    time.sleep(2)
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    print("Generating learning rate at 0.001")
    time.sleep(2)
    learning_rate = 0.001

    for epoch in range(1000):
        for batch in minibatches(inputs, batch_size=20):
            grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
            theta = gradient_step(theta, grad, -learning_rate)
            print(f"epoch: {epoch}, theta: {theta}")

    slope, intercept = theta
    print(f"Final slope at {slope} and intercept at {intercept}.\n Expected --> slope: 10 intercept: 7")
    assert 9.99 < slope < 10.11, "slope should be about 10"
    assert 6.9 < intercept < 7.1,"intercept should be about 7"

if __name__ == "__main__":
    run()
