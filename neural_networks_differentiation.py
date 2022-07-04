import numpy as np
# from scipy.io import loadmat
from math import e


def Initialise(a, b):
    ep = 0.15
    c = np.random.rand(a, b) * (2 * ep) - ep
    return c


def sigmoid(z):
    c = 1 / (1 + pow(e, -z))
    return c


def forward_prop(X, Input_Layer_size, Hidden_Layer_size, Output_Layer_size, weights, biases):
    Theta1 = np.reshape(weights[:Hidden_Layer_size * Input_Layer_size],
                        (Hidden_Layer_size, Input_Layer_size))
    Theta2 = np.reshape(weights[Hidden_Layer_size * Input_Layer_size:],
                        (Output_Layer_size, Hidden_Layer_size))

    # print('theta1: ', Theta1)
    # print('theta2: ', Theta2)
    bias1 = biases[0]
    bias2 = biases[1]

    zH = np.dot(Theta1, X) + bias1
    H = sigmoid(zH)
    zO = np.dot(Theta2, H) + bias2
    # print('zO: ', zO)
    O = sigmoid(zO)

    return_list = [H, O]
    return return_list


def loss(O, T):
    E = 0
    # print('len(T):', T.shape)
    # print('len(O):', O.shape)
    for i in range(len(T) - 1):
        E += pow((O[i] - T[i]), 2)
    return E / 2


def differentiation(X, H, O, T, Theta1, Theta2):
    # dE/dO1 and dE/dO2 (2x1)
    dO = O - T
    dO = dO[np.newaxis]
    # print('dO:',dO)
    # dO1/dzO1 and dO2/dzO2 (2x1)
    dzO = O * (1 - O)
    dzO = dzO[np.newaxis]
    # print('dzO:', dzO)
    # dH1/dzH1 and dH2/dzH2 (2x1)
    dzH = H * (1 - H)
    dzH = dzH[np.newaxis]
    # print('dzH:', dzH)
    # dzO1/dH1 and dzO2/dH1 and dzO1/dH2 and dzO2/dH2 is Theta2 (4x1)

    # dE/dH1 and dE/dH2
    dH = np.dot(dO * dzO, Theta2)
    # print('dH:', dH)

    # dzO1/dw7, dzO2/dw8, dzO1/dw9 and dzO2/dw10
    dz_Theta2 = []
    for i in range(len(Theta2[:, 0])):
        dz_Theta2.append(H)
    dz_Theta2 = np.reshape(dz_Theta2,
                           Theta2.shape)
    # print('dz_Theta2:', dz_Theta2)

    # dE/dw7, dE/dw8, dE/dw9 and dE/dw10
    dTheta2 = (dO * dzO).T * dz_Theta2
    # print('dTheta2: ', dTheta2)

    # dzh1/dw1, dzh1/dw3, dzh1/dw5, dzh2/dw2, dzh2/dw4 and dzh2/dw6
    dz_Theta1 = []
    for i in range(len(Theta1[:, 0])):
        dz_Theta1.append(X)
    dz_Theta1 = np.reshape(dz_Theta1, Theta1.shape)
    # print('dz_Theta1: ', dz_Theta1)

    # dE/dw1, dE/dw2, dE/dw3, dE/dw4, dE/dw5 and dE/dw6
    dTheta1 = (dH * dzH).T * dz_Theta1
    # print('dTheta1: ', dTheta1)

    change_in_weights = np.concatenate((dTheta1.flatten(), dTheta2.flatten()))
    # print('change in weights: ', change_in_weights)

    # dzO1/db2, dzO2/db2,
    dzb2 = np.ones((len(T)))
    # print('dzb2: ',dzb2)

    # dzH1/db1, dzH2/db1
    dzb1 = np.ones((len(H)))
    # print('dzb1: ',dzb1)

    # dE/db2
    db2 = np.sum(dO * dzO * dzb2)
    # print('db2: ', db2)

    # dE/db1
    db1 = np.sum(dO * dzO * np.dot(Theta2, (dzH * dzb1).T))
    # print('db1: ', db1)

    change_in_biases = np.array([db1, db2])

    return change_in_weights, change_in_biases


def new_weights(Theta1, Theta2, biases, change_in_weights, change_in_biases, learning_rate):
    weights = np.concatenate((Theta1.flatten(), Theta2.flatten()))
    # print('weights: ', weights)
    # print('biases: ', biases)

    weights = weights - (change_in_weights * learning_rate)
    new_biases = biases - (change_in_biases * learning_rate)

    # print('new weights: ', new_weights)
    # print('new biases: ', new_biases)

    return weights, new_biases


def minimise(X, T, Input_Layer_size, Hidden_Layer_size, Output_Layer_size,
             initial_weights, biases, learning_rate, num_iter):
    initial_Theta1 = np.reshape(initial_weights[:Hidden_Layer_size * Input_Layer_size],
                                (Hidden_Layer_size, Input_Layer_size))
    initial_Theta2 = np.reshape(initial_weights[Hidden_Layer_size * Input_Layer_size:],
                                (Output_Layer_size, Hidden_Layer_size))

    H, O = forward_prop(X, Input_Layer_size, Hidden_Layer_size, Output_Layer_size, initial_weights, biases)
    # print('H: ', H)
    # print('O: ', O)

    E = loss(O, T)
    # print('loss: ', E)

    change_in_weights, change_in_biases = differentiation(X, H, O, T, initial_Theta1, initial_Theta2)
    # print('change in weights: ', change_in_weights)
    # print('change in biases: ', change_in_biases)

    weights, new_biases = new_weights(initial_Theta1, initial_Theta2, biases, change_in_weights, change_in_biases,
                                      learning_rate)

    losses = [[E, weights, new_biases]]

    for i in range(num_iter):
        print('Iteration ', i + 1)
        H, O = forward_prop(X, Input_Layer_size, Hidden_Layer_size, Output_Layer_size, weights, new_biases)
        E = loss(O, T)

        Theta1 = np.reshape(weights[:Hidden_Layer_size * Input_Layer_size],
                            (Hidden_Layer_size, Input_Layer_size))
        Theta2 = np.reshape(weights[Hidden_Layer_size * Input_Layer_size:],
                            (Output_Layer_size, Hidden_Layer_size))

        change_in_weights, change_in_biases = differentiation(X, H, O, T, Theta1, Theta2)
        weights, new_biases = new_weights(Theta1, Theta2, new_biases, change_in_weights, change_in_biases,
                                          learning_rate)
        losses.append([E, weights, new_biases])

        print('Loss after iteration is ', E)

    min_E = losses[0][0]
    index = 0
    for i in range(len(losses)):
        if losses[i][0] < min_E:
            min_E = losses[i][0]
            index = i

    weights = losses[index][1]
    new_biases = losses[index][2]

    return min_E, weights, new_biases
