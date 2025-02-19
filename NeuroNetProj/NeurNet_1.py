import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_purp):
    return ((y_true-y_purp)**2).mean()


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, x):
        total = np.dot(self.weights, x) + self.bias
        sig_total = sigmoid(total)
        return [total, sig_total]


class NeuralNetwork:
    def __init__(self):
        weights = [np.random.normal() for _ in range(10)]
        bias = [np.random.normal() for _ in range(3)]
        self.h1 = Neuron(weights[0:2], bias[0])
        self.h2 = Neuron(weights[2:4], bias[1])
        self.o1 = Neuron(weights[4:6], bias[2])

    def feedforward(self, x):
        h1 = self.h1.feedforward(x)
        h2 = self.h2.feedforward(x)
        o1 = self.o1.feedforward([h1[1], h2[1]])

        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                h1 = self.h1.feedforward(x)
                h2 = self.h2.feedforward(x)
                o1 = self.o1.feedforward([h1[1], h2[1]])
                y_pred = o1[1]
                # Нейрон ol
                d_L_d_ypred = -2*(y_true - y_pred)
                d_ypred_d_w5 = h1[1] * deriv_sigmoid(o1[0])
                d_ypred_d_w6 = h2[1] * deriv_sigmoid(o1[0])
                d_ypred_d_b3 = deriv_sigmoid(o1[0])

                d_ypred_d_h1 = self.o1.weights[0] * deriv_sigmoid(o1[0])
                d_ypred_d_h2 = self.o1.weights[1] * deriv_sigmoid(o1[0])
                # Нейрон hl
                d_h1_d_w1 = x[0] * deriv_sigmoid(h1[0])
                d_h1_d_w2 = x[1] * deriv_sigmoid(h1[0])
                d_h1_d_b1 = deriv_sigmoid(h1[0])
                # Нейрон h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(h2[0])
                d_h2_d_w4 = x[1] * deriv_sigmoid(h2[0])
                d_h2_d_b2 = deriv_sigmoid(h2[0])
                # --- Обновляем вес и смещения
                # Нейрон hl
                self.h1.weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 \
                    * d_h1_d_w1
                self.h1.weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 \
                    * d_h1_d_w2
                self.h1.bias -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * \
                    d_h1_d_b1
                # Нейрон h2
                self.h2.weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 \
                    * d_h2_d_w3
                self.h2.weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 \
                    * d_h2_d_w4
                self.h2.bias -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * \
                    d_h2_d_b2
                # Нейрон o1
                self.o1.weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.o1.weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.o1.bias -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            if epoch % 100 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                first_elements = []
                for row in y_preds:
                    first_elements.append(row[1])
                loss = mse_loss(all_y_trues, first_elements)
                print("Epoch %d loss: %.3f" % (epoch, loss))


data = np.array([
    [-2, -1],   # alice
    [25, 6],    # bob
    [17, 4],    # charlie
    [-15, -6],  # diana
])

all_y_trues = np.array([
    1,      # alice
    0,      # bob
    0,      # charlie
    1,      # diana
])

network = NeuralNetwork()
network.train(data, all_y_trues)
emily = np.array([-7, -3])
frank = np.array([20, 2])
print("Emily: %.3f" % network.feedforward(emily)[1])
print("Frank: %.3f" % network.feedforward(frank)[1])
