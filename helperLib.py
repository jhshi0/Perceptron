import numpy, math, datetime

from datetime import datetime

from random import randint


def read_and_sort_file_into_datasets(filepath):

    return __read_file_into_output(filepath)


def compute_eucleadian_norm(x, y, z):

    return math.sqrt(x**2 + y**2 + z**2)


def heaviside(input):

    return 0.5 * (numpy.sign(input) + 1)


def __read_file_into_output(filepath):

    with open(filepath) as reader:

        all_lines = reader.readlines()

        len_batch = len(all_lines) - 1

        labels = numpy.zeros((1, len_batch))

        features = {}

        for i in range(1, 3):
            features[i] = numpy.zeros((1, len_batch))

        for i in range(len(all_lines)):
            if i == 0:
                pass

            else:
                tokens = all_lines[i].split()

                labels[0, i-1] = tokens[0]

                for m in range(1, 3):
                    features[m][0, i-1] = tokens[m]

    assert isinstance(labels, numpy.ndarray)
    assert isinstance(features[1], numpy.ndarray)
    assert isinstance(features[2], numpy.ndarray)

    return (labels, features[1], features[2])


def write_input_and_save_filenames():

    output =\
    {
        "smalldataset": "/Users/jhshi/PycharmProjects/"
                         "Perceptron/data_set1",
        "largedataset": "/Users/jhshi/PycharmProjects/"
                        "Perceptron/data_set2",
        "savepath1": "/Users/jhshi/Desktop/PercFiles/"
                     + str(datetime.now().strftime("%H-%M-%S"))
                     + "-" + str(randint(0, 100)),
        "savepath2": "/Users/jhshi/Desktop/PercFiles/"
                    + str(datetime.now().strftime("%H-%M-%S"))
                    + "-" + str(randint(0, 100)),
        "savepath3": "/Users/jhshi/Desktop/PercFiles/"
                    + str(datetime.now().strftime("%H-%M-%S"))
                    + "-" + str(randint(0, 100))
    }

    return output


def generate_range_rand_ints(start, finish, num_items):

    output = []
    output_ctr = Iterator()
    isEnd = False

    while isEnd != True:

        m = randint(start, finish)

        if m != 0:
            output.append(m)
        else:
            pass

        if len(output) == num_items:
            isEnd = True
        else:
            pass

        output_ctr.increment()

    assert isinstance(start, int)
    assert isinstance(finish, int)
    assert isinstance(num_items, int)
    assert len(output) == num_items

    return output


def retrieve_last_item(input):

    assert isinstance(input, list)

    l = len(input)

    return input[l-1]


def sort_data_by_labels(labels, feature1, feature2):

    x1y0 = []
    x2y0 = []
    x1y1 = []
    x2y1 = []

    s = numpy.shape(labels)
    l = s[1]

    for i in range(l):
        if labels[0, i] == 0:
            x1y0.append(feature1[0, i])
            x2y0.append(feature2[0, i])

        elif labels[0, i] == 1:
            x1y1.append(feature1[0, i])
            x2y1.append(feature2[0, i])

        else:
            raise Exception("label not 0 or 1")

    return (x1y0, x2y0, x1y1, x2y1)


def generate_linear_range(start, end, num_points):

    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(num_points, int)

    linrange = []

    lowerbound = start; upperbound = end
    step = math.floor((upperbound - lowerbound) / num_points)
    for i in range(num_points+1):
        linrange.append(lowerbound + step * i)

    if upperbound not in linrange:
        linrange.append(upperbound)

    return linrange


class Iterator:

    def __init__(self):

        self.counter = int(0)

    def increment(self):

        self.counter += 1

    def decrement(self):

        self.counter -= 1

    def reset(self):

        self.counter = int(0)

    def read(self):

        return self.counter


class ModelTester:

    def __init__(self):

        self.theta = {}

        for i in range(0, 3):
            self.theta[i] = None

        self.test_x = {}

        for j in range(0, 3):
            self.test_x[j] = None

        self.test_y = None

        self.J_score = None

    def prompt_model(self, theta0, theta1, theta2):

        self.theta[0] = theta0
        self.theta[1] = theta1
        self.theta[2] = theta2

        return

    def prompt_testset(self, x0, x1, x2, labels):

        self.test_x[0] = x0
        self.test_x[1] = x1
        self.test_x[2] = x2
        self.test_y = labels

        return

    def initialize_computational_attributes(self):

        self.testset_shape = numpy.shape(self.test_x[0])

        return

    def evaluate_J_score(self):

        self.initialize_computational_attributes()

        eta = numpy.zeros(self.testset_shape)

        for m in range(0, 3):
            eta = eta + self.test_x[m] * self.theta[m]

        error = self.test_y - heaviside(eta)

        self.J_score = numpy.inner(error, error)[0, 0]

        return

    def __str__(self):

        return "J=" + str(self.J_score)
