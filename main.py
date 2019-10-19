import mini_batches
import stochastic
import gradient_descent


def choose_gradient():
    print("*********************************")
    print("*******Choose your gradient technique*******")
    print("*********************************")

    print("(1) Gradient Descent (2) Mini-batches (3) Stochastic")

    technique = int(input("Which one?"))


    if technique == 1:
        print("Starting Gradient Descent....")
        gradient_descent.run()
    elif technique == 2:
        print("Starting Mini-batches....")
        mini_batches.run()
    elif technique == 3:
        print("Starting Stochastic...")
        stochastic.run()
    else:
        print("Something went wrong")

if __name__ == "__main__":
    choose_gradient()
