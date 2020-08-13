import numpy, math

from helperLib import compute_eucleadian_norm, Iterator, heaviside
from itertools import product

class StochyPerc:

    def __init__(self):

        self.input = {}

    def prompt_input(self, labels, feature1, feature2, feature0):

        self.y = labels

        self.x = {}

        self.x[0] = feature0
        self.x[1] = feature1
        self.x[2] = feature2

        return

    def prompt_parameters(
        self, theta0, theta1, theta2, num_epochs, training_rate):

        self.theta = {}

        self.theta[0]   = theta0
        self.theta[1]   = theta1
        self.theta[2]   = theta2

        self.num_epochs = num_epochs
        self.alpha      = training_rate

        return

    def optimize_params(self):

        self.__define_and_initialize_computational_attributes()
        self.__run_optimizer()

        return

    def __define_and_initialize_computational_attributes(self):

        self.grad = {}

        for i in range(0, 3):
            self.grad[i] = 0.0

        self.error = None
        self.eta   = None

        self.J_score     = 0.0
        self.pile_thetas = []

        return

    def __run_optimizer(self):

        self.loop_ctr = Iterator()
        self.is_optimized = False

        r = product(range(self.num_epochs),
                    range(numpy.shape(self.y)[1]))

        self.__optimize_params_via_for_loop(range_pair=r)

        return

    def __optimize_params_via_for_loop(self, range_pair):
        for epochs, batch_index in range_pair:

            if self.is_optimized:
                break
            else:
                pass

            self.__save_thetas_to_pile()
            self.__compute_gradient_and_update_thetas(
                batch_index, heaviside)

            self.__update_J_score()
            self.__determine_if_optimization_end()

            self.loop_ctr.increment()

    def __compute_gradient_and_update_thetas(self, batch_index, activation):

        self.eta = numpy.zeros(numpy.shape(self.y))

        norm_theta = compute_eucleadian_norm(
            self.theta[0], self.theta[1], self.theta[2])

        for i in range(0, 3):
            self.eta = self.eta + self.theta[i] * self.x[i]

        self.error = self.y - activation(self.eta/norm_theta)

        err_t = numpy.transpose(self.error)

        for j in range(0, 3):
            self.grad[j] = err_t[batch_index, 0] \
                         * self.x[j][0, batch_index]

        for k in range(0, 3):
            self.theta[k] = self.theta[k] \
                          + self.alpha * self.grad[k]

        return

    def __save_thetas_to_pile(self):

        a = (self.theta[0], self.theta[1], self.theta[2])
        self.pile_thetas.append(a)

        return

    def __update_J_score(self):

        self.J_score = self.__calculate_J_score()

        return

    def __determine_if_optimization_end(self):

        if self.J_score <= 0.01:
            self.is_optimized = True

        else:
            pass

        return

    def __calculate_J_score(self):

        m = numpy.dot(
                       self.error,
                       numpy.transpose(self.error))

        return m[0,0]

    def get_from_input(self, keyword):

        return self.input[keyword]



