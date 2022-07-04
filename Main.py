import numpy as np
import neural_networks_differentiation as nnd


def main():
    X = np.array([1, 2, 4])
    T = np.array([0.1, 0.05])

    Input_Layer_size = 3
    Hidden_Layer_size = 2
    Output_Layer_size = 2
    num_iter = 100
    learning_rate = 0.7

    # Randomly initialising Thetas
    initial_Theta1 = nnd.Initialise(Hidden_Layer_size, Input_Layer_size)
    initial_Theta2 = nnd.Initialise(Output_Layer_size, Hidden_Layer_size)
    # print('initial_theta1: ', initial_Theta1)
    # print('initial_theta2: ', initial_Theta2)

    initial_weights = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
    initial_bias1 = 1
    initial_bias2 = 1
    biases = np.array([initial_bias1, initial_bias2])

    min_E, weights, new_biases = nnd.minimise(X, T, Input_Layer_size, Hidden_Layer_size, Output_Layer_size,
                                              initial_weights, biases, learning_rate, num_iter)

    print('\n\nITERATIONS COMPLETE')
    print('\nminimum loss is: ', min_E,
          # '\non iteration: ',index,
          '\nnew weights: ', weights,
          '\nnew biases: ', new_biases)

    _, pred = nnd.forward_prop([1, 2, 4], Input_Layer_size, Hidden_Layer_size, Output_Layer_size, weights, new_biases)

    print('predicted output is: ', pred)


main()
