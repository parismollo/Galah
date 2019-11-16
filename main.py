import mini_batches
import stochastic
import gradient_descent
import time


def choose_gradient():
    print("********************************************")
    print("*******Choose your gradient technique*******")
    print("*******************Linear*******************")
    print("********************************************")

    print("(1): Gradient Descent (2): Mini-batches (3): Stochastic")
    technique = 0
    while technique not in range(1, 4):
        technique = int(input("Which one? "))

        if technique == 1:
            print("Starting Gradient Descent....")
            time.sleep(5)
            gradient_descent.run()
        elif technique == 2:
            print("Starting Mini-batches....")
            time.sleep(5)
            mini_batches.run()
        elif technique == 3:
            print("Starting Stochastic...")
            time.sleep(5)
            stochastic.run()
        else:
            print("Choose a valid number")


if __name__ == "__main__":
    choose_gradient()
