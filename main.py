import numpy, matplotlib

import matplotlib.pyplot as matplot

from random import randint

from datetime import datetime

from numpy import linspace

from math import floor

from matplotlib.ticker import MaxNLocator

from regresionLib import StochyPerc

from helperLib \
    import \
    read_and_sort_file_into_datasets,\
    write_input_and_save_filenames, \
    generate_range_rand_ints,\
    retrieve_last_item,\
    Iterator,\
    sort_data_by_labels,\
    ModelTester,\
    generate_linear_range

from itertools import chain

def plot_datasets_and_multiple_decision_boundaries():

    exportfiles = write_input_and_save_filenames()

    datasets = read_and_sort_file_into_datasets(
        exportfiles["smalldataset"])

    labels = datasets[0]
    x1 = datasets[1]
    x2 = datasets[2]

    sorted_datasets = sort_data_by_labels(
        feature2=x2, feature1=x1, labels=labels)

    x1y0 = sorted_datasets[0]
    x2y0 = sorted_datasets[1]
    x1y1 = sorted_datasets[2]
    x2y1 = sorted_datasets[3]

    x0 = numpy.ones(numpy.shape(labels))
    batch_size = numpy.shape(labels)[1]

    models = StochyPerc()

    models.prompt_input(
        labels=datasets[0], feature1=datasets[1], feature2=datasets[2],
        feature0=x0)

    num_models_trained = 500

    upper_bound_th0_init = int(1000)
    lower_bound_th0_init = int(-1000)

    rand_range = generate_range_rand_ints(
        start=lower_bound_th0_init,
        finish=upper_bound_th0_init,
        num_items=num_models_trained)

    set_slope = []; set_intercept = []

    counter_success = Iterator()

    for j in rand_range:

        models.prompt_parameters(
            theta0=j, theta2=1, theta1=1,
            num_epochs=3000,
            training_rate=0.5)

        models.optimize_params()

        if models.J_score == 0:
            l = len(models.pile_thetas)
            pile = models.pile_thetas[l - 1]
            set_slope.append(-1 * pile[1] / pile[2])
            set_intercept.append(-1 * pile[0] / pile[2])
            counter_success.increment()
            print counter_success.read()
        else:
            pass

    plot_title_model_parameters = (
        str(counter_success.read())
        + "/"
        + str(num_models_trained)
        + " "
        + "successes"
        + " "
        + r"$\alpha=$" + str(models.alpha)
        + " "
        + r"updates"
        + r"$\leq$"
        + str(models.num_epochs)
        + "\n"
        + " Initializations:"
        + r"$\theta_{1},\theta_{2}=$"
        + str(models.pile_thetas[0][1])
        + ", "
        + r"$\theta_{0} \in$"
        + r"$[$"
        + str(lower_bound_th0_init)
        + r"$,$"
        + str(upper_bound_th0_init)
        + r"$]$")

    matplot.figure(2)
    matplot.xlabel(r"$x_{1}$")
    matplot.ylabel(r"$x_{2}$")
    matplot.title(plot_title_model_parameters)

    matplot.scatter(x1y0, x2y0, color='indigo', zorder=100)
    matplot.scatter(x1y1, x2y1, color='lawngreen', zorder=100)

    horizon_min = numpy.amin(x1)
    horizon_max = numpy.amax(x1)
    vertical_min = numpy.amin(x2)
    vertical_max = numpy.amax(x2)

    for q in range(len(set_slope)):
        slope = set_slope[q]
        intercept = set_intercept[q]
        y_min = horizon_min * slope + intercept
        y_max = horizon_max * slope + intercept

        matplot.plot(
            [horizon_min, horizon_max],
            [y_min, y_max],
            linewidth=1,
            zorder=1,
            color='grey')

    matplot.xlim([horizon_min, horizon_max])
    matplot.ylim([vertical_min, vertical_max])

    matplot.savefig(exportfiles["savepath2"])
    matplot.close(2)

    return

def plot_converged_models_over_epoches_varying_th0():

    models = StochyPerc()

    exportfiles = write_input_and_save_filenames()

    datasets = read_and_sort_file_into_datasets(
        exportfiles["smalldataset"])

    labels = datasets[0]

    x0 = numpy.ones(numpy.shape(labels))
    batch_size = numpy.shape(labels)[1]

    models.prompt_input(
        labels=datasets[0], feature1=datasets[1], feature2=datasets[2],
        feature0=x0)

    num_models_trained = 500
    upper_bound_th0_init = int(500)
    lower_bound_th0_init = int(-500)

    rand_range = generate_range_rand_ints(
        start=lower_bound_th0_init,
        finish=upper_bound_th0_init,
        num_items=num_models_trained)

    all_updates = []; convergences = []

    counter_success = Iterator()

    linrange = []
    for m in range(1, 31):
        linrange.append(m * 100)

    for max_updates in linrange:
        print str(max_updates)
        for i in rand_range:

            models.prompt_parameters(theta0=i, theta2=1, theta1=1,
                                     num_epochs=max_updates,
                                     training_rate=0.5)

            models.optimize_params()

            if models.J_score == 0:
                counter_success.increment()
            else:
                pass

        convergences.append(
            counter_success.read())

        all_updates.append(max_updates)

        counter_success.reset()

    title_model_parameters = (
        str(num_models_trained)
        + "models trained"
        + ", "
        + r"$\alpha=$"
        + str(models.alpha)
        + "\n"
        + "Initializations: "
        + r"$\theta_{0}, \theta_{1}=$"
        + str(models.pile_thetas[0][0])
        + " , "
        + r"$\theta_{0} \in$"
        + "["
        + str(lower_bound_th0_init)
        + ","
        + str(upper_bound_th0_init)
        + "]")

    fig = matplot.figure(2)
    axs = fig.add_subplot(111)
    axs.set_xlabel("epoches")
    axs.set_ylabel("models converged")
    fig.suptitle(title_model_parameters)

    axs.scatter(all_updates, convergences, color="black")
    x_min = 0; y_min = 0
    x_max = max(all_updates) + 1
    y_max = max(convergences) + 1

    axs.set_xlim([x_min, x_max])
    axs.set_ylim([y_min, y_max])
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)

    axs.xaxis.set_major_locator(
        MaxNLocator(integer=True))
    axs.yaxis.set_major_locator(
        MaxNLocator(integer=True))

    fig.savefig(exportfiles["savepath1"])
    matplot.close(2)

    return

def plot_converged_models_over_epoches_varying_th1():

    models = StochyPerc()
    exportfiles = write_input_and_save_filenames()
    datasets = read_and_sort_file_into_datasets(
        exportfiles["smalldataset"])

    labels = datasets[0]

    x0 = numpy.ones(numpy.shape(labels))
    batch_size = numpy.shape(labels)[1]

    models.prompt_input(
        labels=datasets[0], feature1=datasets[1], feature2=datasets[2],
        feature0=x0)

    num_models_trained = 500
    upper_bound_th1_init = int(500)
    lower_bound_th1_init = int(-500)

    rand_range = generate_range_rand_ints(
        start=lower_bound_th1_init,
        finish=upper_bound_th1_init,
        num_items=num_models_trained)

    all_updates = []; convergences = []

    counter_success = Iterator()

    linrange = []
    for m in range(1, 31):
        linrange.append(m * 100)

    for max_updates in linrange:
        print str(max_updates)
        for i in rand_range:

            models.prompt_parameters(theta0=1, theta2=1, theta1=i,
                                     num_epochs=max_updates,
                                     training_rate=0.5)

            models.optimize_params()

            if models.J_score == 0:
                counter_success.increment()
            else:
                pass

        convergences.append(
            counter_success.read())

        all_updates.append(max_updates)

        counter_success.reset()

    title_model_parameters = (
        str(num_models_trained)
        + "models trained"
        + ", "
        + r"$\alpha=$"
        + str(models.alpha)
        + "\n"
        + "Initializations: "
        + r"$\theta_{0}, \theta_{2}=$"
        + str(models.pile_thetas[0][2])
        + " , "
        + r"$\theta_{1} \in$"
        + "["
        + str(lower_bound_th1_init)
        + ","
        + str(upper_bound_th1_init)
        + "]")

    fig = matplot.figure(2)
    axs = fig.add_subplot(111)
    axs.set_xlabel("epoches")
    axs.set_ylabel("models converged")
    fig.suptitle(title_model_parameters)

    axs.scatter(all_updates, convergences, color="black")
    x_min = 0; y_min = 0
    x_max = max(all_updates) + 1
    y_max = max(convergences) + 1

    axs.set_xlim([x_min, x_max])
    axs.set_ylim([y_min, y_max])
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)

    axs.xaxis.set_major_locator(
        MaxNLocator(integer=True))
    axs.yaxis.set_major_locator(
        MaxNLocator(integer=True))

    fig.savefig(exportfiles["savepath1"])
    matplot.close(2)

    return

def plot_converged_models_over_epoches_varying_th2():

    models = StochyPerc()
    exportfiles = write_input_and_save_filenames()
    datasets = read_and_sort_file_into_datasets(
        exportfiles["smalldataset"])

    labels = datasets[0]

    x0 = numpy.ones(numpy.shape(labels))
    batch_size = numpy.shape(labels)[1]

    models.prompt_input(
        labels=datasets[0], feature1=datasets[1], feature2=datasets[2],
        feature0=x0)

    num_models_trained = 500
    upper_bound_th2_init = int(500)
    lower_bound_th2_init = int(-500)

    rand_range = generate_range_rand_ints(
        start=lower_bound_th2_init,
        finish=upper_bound_th2_init,
        num_items=num_models_trained)

    all_updates = []; convergences = []

    counter_success = Iterator()

    linrange = []
    for m in range(1, 31):
        linrange.append(m * 100)

    for max_updates in linrange:
        print str(max_updates)
        for i in rand_range:

            models.prompt_parameters(theta0=1, theta2=i, theta1=1,
                                     num_epochs=max_updates,
                                     training_rate=0.5)

            models.optimize_params()

            if models.J_score == 0:
                counter_success.increment()
            else:
                pass

        convergences.append(
            counter_success.read())

        all_updates.append(max_updates)

        counter_success.reset()

    title_model_parameters = (
        str(num_models_trained)
        + "models trained"
        + ", "
        + r"$\alpha=$"
        + str(models.alpha)
        + "\n"
        + "Initializations: "
        + r"$\theta_{0}, \theta_{1}=$"
        + str(models.pile_thetas[0][0])
        + " , "
        + r"$\theta_{2} \in$"
        + "["
        + str(lower_bound_th2_init)
        + ","
        + str(upper_bound_th2_init)
        + "]")

    fig = matplot.figure(2)
    axs = fig.add_subplot(111)
    axs.set_xlabel("epoches")
    axs.set_ylabel("models converged")
    fig.suptitle(title_model_parameters)

    axs.scatter(all_updates, convergences, color="black")
    x_min = 0; y_min = 0
    x_max = max(all_updates) + 1
    y_max = max(convergences) + 1

    axs.set_xlim([x_min, x_max])
    axs.set_ylim([y_min, y_max])
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)

    axs.xaxis.set_major_locator(
        MaxNLocator(integer=True))
    axs.yaxis.set_major_locator(
        MaxNLocator(integer=True))

    fig.savefig(exportfiles["savepath1"])
    matplot.close(2)

    return

def plot_J_scores_over_initializations():

    models = StochyPerc()
    tester = ModelTester()

    exportfiles = write_input_and_save_filenames()
    testing_set = read_and_sort_file_into_datasets(
        exportfiles["largedataset"])
    training_set = read_and_sort_file_into_datasets(
        exportfiles["smalldataset"])

    testset_x = {}

    training_labels = training_set[0]

    model_x0 = numpy.ones(numpy.shape(training_labels))
    tester_x0 = numpy.ones(numpy.shape(testing_set[0]))

    models.prompt_input(
        labels=training_labels,
        feature1=training_set[1],
        feature2=training_set[2],
        feature0=model_x0)

    training_rate = 0.5; upperbound_epoches = 3000
    min_th0 = -1000; max_th0 = 1000; num_models = 500

    rand_range = generate_range_rand_ints(
        min_th0, max_th0, num_models)

    init_thetas = []; fin_thetas = []

    linrange = generate_linear_range(-1000, 100, num_models)

    success_ctr = Iterator()

    for i in rand_range:
        models.prompt_parameters(
            theta0=i, theta1=1, theta2=1,
            num_epochs=upperbound_epoches,
            training_rate=training_rate)

        models.optimize_params()

        if models.J_score == 0:
            init_thetas.append(
                models.pile_thetas[0])
            fin_thetas.append(
                models.pile_thetas[-1])
            success_ctr.increment()
            print success_ctr.read()
        else:
            pass

    print init_thetas
    # print fin_thetas

    scorings = []; testset_init_th0 = []

    for j in range(len(fin_thetas)):
        optimized_params = fin_thetas[j]

        tester.prompt_model(
            theta0=optimized_params[0],
            theta1=optimized_params[1],
            theta2=optimized_params[2])

        tester.prompt_testset(
            x0=tester_x0,
            x1=testing_set[1],
            x2=testing_set[2],
            labels=testing_set[0])

        tester.evaluate_J_score()
        scorings.append(tester.J_score)
        testset_init_th0.append(init_thetas[j][0])

    print scorings
    print testset_init_th0

    title_params = (
        r"$\alpha=$"
        + "%.2f" % training_rate
        + " "
        + "epoches"
        + r"$\leq$"
        + str(upperbound_epoches)
        + "\n"
        + str(success_ctr.read())
        + "/"
        + str(len(rand_range))
        + " "
        + "model(s)")

    fig = matplot.figure(3)
    axs = matplot.subplot(111)
    fig.suptitle(title_params)

    axs.scatter(testset_init_th0, scorings,
                color="black")

    axs.set_xlabel(r"$\theta_{0}$")
    axs.set_ylabel(r"$J$")

    x_min = min(testset_init_th0)
    x_max = max(testset_init_th0)
    y_min = 0
    y_max = max(scorings) + 1

    axs.set_xlim([x_min, x_max])
    axs.set_ylim([y_min, y_max])

    axs.xaxis.set_major_locator(
        MaxNLocator(integer=True))
    axs.yaxis.set_major_locator(
        MaxNLocator(integer=True))

    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)

    matplot.savefig(exportfiles["savepath3"])
    matplot.close(3)

    return

def plot_datasets_and_bounds_varying_theta1():
    exportfiles = write_input_and_save_filenames()

    datasets = read_and_sort_file_into_datasets(
        exportfiles["smalldataset"])

    labels = datasets[0]
    x1 = datasets[1]
    x2 = datasets[2]

    sorted_datasets = sort_data_by_labels(
        feature2=x2, feature1=x1, labels=labels)

    x1y0 = sorted_datasets[0]
    x2y0 = sorted_datasets[1]
    x1y1 = sorted_datasets[2]
    x2y1 = sorted_datasets[3]

    x0 = numpy.ones(numpy.shape(labels))
    batch_size = numpy.shape(labels)[1]

    models = StochyPerc()

    models.prompt_input(
        labels=datasets[0], feature1=datasets[1],
        feature2=datasets[2], feature0=x0)

    num_models_trained = 500

    upper_bound_th1_init = int(1000)
    lower_bound_th1_init = int(-1000)

    rand_range = generate_range_rand_ints(
        start=lower_bound_th1_init,
        finish=upper_bound_th1_init,
        num_items=num_models_trained)

    set_slope = []; set_intercept = []

    counter_success = Iterator()

    print rand_range

    for j in rand_range:

        models.prompt_parameters(
            theta0=1, theta2=1, theta1=j,
            num_epochs=3000,
            training_rate=0.5)

        models.optimize_params()

        if models.J_score == 0:
            l = len(models.pile_thetas)
            pile = models.pile_thetas[l - 1]
            set_slope.append(-1 * pile[1] / pile[2])
            set_intercept.append(-1 * pile[0] / pile[2])
            counter_success.increment()
            print counter_success.read()
        else:
            pass

    plot_title_model_parameters = (
        str(counter_success.read())
        + "/"
        + str(num_models_trained)
        + " "
        + "successes"
        + " "
        + r"$\alpha=$" + str(models.alpha)
        + " "
        + r"updates"
        + r"$\leq$"
        + str(models.num_epochs)
        + "\n"
        + " Initializations:"
        + r"$\theta_{0},\theta_{2}=$"
        + str(models.pile_thetas[0][0])
        + ", "
        + r"$\theta_{1} \in$"
        + r"$[$"
        + str(lower_bound_th1_init)
        + r"$,$"
        + str(upper_bound_th1_init)
        + r"$]$")

    matplot.figure(2)
    matplot.xlabel(r"$x_{1}$")
    matplot.ylabel(r"$x_{2}$")
    matplot.title(plot_title_model_parameters)

    matplot.scatter(x1y0, x2y0, color='indigo', zorder=100)
    matplot.scatter(x1y1, x2y1, color='lawngreen', zorder=100)

    horizon_min = numpy.amin(x1)
    horizon_max = numpy.amax(x1)
    vertical_min = numpy.amin(x2)
    vertical_max = numpy.amax(x2)

    for q in range(len(set_slope)):
        slope = set_slope[q]
        intercept = set_intercept[q]
        y_min = horizon_min * slope + intercept
        y_max = horizon_max * slope + intercept

        matplot.plot(
            [horizon_min, horizon_max],
            [y_min, y_max],
            linewidth=1,
            zorder=1,
            color='grey')

    matplot.xlim([horizon_min, horizon_max])
    matplot.ylim([vertical_min, vertical_max])

    matplot.savefig(exportfiles["savepath2"])
    matplot.close(2)

    return

def plot_datasets_and_bounds_varying_theta2():
    exportfiles = write_input_and_save_filenames()

    datasets = read_and_sort_file_into_datasets(
        exportfiles["smalldataset"])

    labels = datasets[0]
    x1 = datasets[1]
    x2 = datasets[2]

    sorted_datasets = sort_data_by_labels(
        feature2=x2, feature1=x1, labels=labels)

    x1y0 = sorted_datasets[0]
    x2y0 = sorted_datasets[1]
    x1y1 = sorted_datasets[2]
    x2y1 = sorted_datasets[3]

    x0 = numpy.ones(numpy.shape(labels))
    batch_size = numpy.shape(labels)[1]

    models = StochyPerc()

    models.prompt_input(
        labels=datasets[0], feature1=datasets[1],
        feature2=datasets[2], feature0=x0)

    num_models_trained = 500

    upper_bound_th1_init = int(1000)
    lower_bound_th1_init = int(-1000)

    rand_range = generate_range_rand_ints(
        start=lower_bound_th1_init,
        finish=upper_bound_th1_init,
        num_items=num_models_trained)

    set_slope = []; set_intercept = []

    counter_success = Iterator()

    print len(rand_range)

    for j in rand_range:

        models.prompt_parameters(
            theta0=1, theta2=j, theta1=1,
            num_epochs=3000,
            training_rate=0.5)

        models.optimize_params()

        if models.J_score == 0:
            l = len(models.pile_thetas)
            pile = models.pile_thetas[l - 1]
            set_slope.append(-1 * pile[1] / pile[2])
            set_intercept.append(-1 * pile[0] / pile[2])
            counter_success.increment()
            print counter_success.read()
        else:
            pass

    plot_title_model_parameters = (
        str(counter_success.read())
        + "/"
        + str(num_models_trained)
        + " "
        + "successes"
        + " "
        + r"$\alpha=$" + str(models.alpha)
        + " "
        + r"updates"
        + r"$\leq$"
        + str(models.num_epochs)
        + "\n"
        + " Initializations:"
        + r"$\theta_{0},\theta_{1}=$"
        + str(models.pile_thetas[0][0])
        + ", "
        + r"$\theta_{2} \in$"
        + r"$[$"
        + str(lower_bound_th1_init)
        + r"$,$"
        + str(upper_bound_th1_init)
        + r"$]$")

    matplot.figure(2)
    matplot.xlabel(r"$x_{1}$")
    matplot.ylabel(r"$x_{2}$")
    matplot.title(plot_title_model_parameters)

    matplot.scatter(x1y0, x2y0, color='indigo', zorder=100)
    matplot.scatter(x1y1, x2y1, color='lawngreen', zorder=100)

    horizon_min = numpy.amin(x1)
    horizon_max = numpy.amax(x1)
    vertical_min = numpy.amin(x2)
    vertical_max = numpy.amax(x2)

    for q in range(len(set_slope)):
        slope = set_slope[q]
        intercept = set_intercept[q]
        y_min = horizon_min * slope + intercept
        y_max = horizon_max * slope + intercept

        matplot.plot(
            [horizon_min, horizon_max],
            [y_min, y_max],
            linewidth=1,
            zorder=1,
            color='grey')

    matplot.xlim([horizon_min, horizon_max])
    matplot.ylim([vertical_min, vertical_max])

    matplot.savefig(exportfiles["savepath2"])
    matplot.close(2)

    return

def test_activation_tanh():

    models = StochyPerc()
    exportfiles = write_input_and_save_filenames()
    datasets = read_and_sort_file_into_datasets(
        exportfiles["smalldataset"])

    labels = datasets[0]; x1 = datasets[1]
    x2 = datasets[2]; x0 = numpy.ones(numpy.shape(labels))

    models.prompt_input(
        labels=datasets[0], feature1=datasets[1],
        feature2=datasets[2], feature0=x0)

    ctr_success = Iterator()

    rand_range = generate_range_rand_ints(-100, 100, 100)

    for i in rand_range:

        models.prompt_parameters(
            theta0=i, theta2=1, theta1=1,
            num_epochs=3000, training_rate=0.01)

        models.optimize_params()

        if models.J_score == 0:
            ctr_success.increment()
            print ctr_success.read()

    for j in rand_range:
        models.prompt_parameters(
            theta0=1, theta2=j, theta1=1,
            num_epochs=3000, training_rate=0.01)

        models.optimize_params()

        if models.J_score == 0:
            ctr_success.increment()
            print ctr_success.read()

    for k in rand_range:
        models.prompt_parameters(
            theta0=1, theta2=1, theta1=k,
            num_epochs=3000, training_rate=0.01)

        models.optimize_params()

        if models.J_score == 0:
            ctr_success.increment()
            print ctr_success.read()

    return

def test_histo_J_scores():

    exportfiles = write_input_and_save_filenames()
    desktop_destination = exportfiles["savepath1"]

    incidences = generate_range_rand_ints(-10, 10, 100)
    bins_edges = generate_linear_range(-11, 11, 10)

    print bins_edges
    print incidences

    num_samples = len(incidences)

    title_J_histogram = (
        str(num_samples)
        + " "
        + "samples")

    fig = matplot.figure()
    axs = matplot.subplot(111)

    fig.suptitle(title_J_histogram)
    axs.set_ylabel("frequency")
    axs.set_xlabel(r"$J$")
    axs.hist(incidences, bins=bins_edges)

    matplot.savefig(desktop_destination)
    matplot.close()

    return

def plot_J_score_variabilities_over_th0():

    models = StochyPerc()
    tester = ModelTester()

    exportfiles = write_input_and_save_filenames()
    testing_set = read_and_sort_file_into_datasets(
        exportfiles["largedataset"])
    training_set = read_and_sort_file_into_datasets(
        exportfiles["smalldataset"])

    testset_x = {}

    training_labels = training_set[0]

    model_x0 = numpy.ones(numpy.shape(training_labels))
    tester_x0 = numpy.ones(numpy.shape(testing_set[0]))

    models.prompt_input(
        labels=training_labels,
        feature1=training_set[1],
        feature2=training_set[2],
        feature0=model_x0)

    training_rate = 0.5; upperbound_epoches = 3000
    min_th0 = -600; max_th0 = -150; num_models = 400

    rand_range = generate_range_rand_ints(
        min_th0, max_th0, num_models)

    init_thetas = []; fin_thetas = []

    linrange = generate_linear_range(-600, 150, num_models)

    success_ctr = Iterator()

    for i in rand_range:
        models.prompt_parameters(
            theta0=i, theta1=1, theta2=1,
            num_epochs=upperbound_epoches,
            training_rate=training_rate)

        models.optimize_params()

        if models.J_score == 0:
            success_ctr.increment()
            fin_thetas.append(models.pile_thetas[-1])
            print success_ctr.read()
        else:
            raise Exception("model not convergent")

    print init_thetas

    scorings = []

    for j in range(len(fin_thetas)):
        optimized_params = fin_thetas[j]

        tester.prompt_model(
            theta0=optimized_params[0],
            theta1=optimized_params[1],
            theta2=optimized_params[2])

        tester.prompt_testset(
            x0=tester_x0,
            x1=testing_set[1],
            x2=testing_set[2],
            labels=testing_set[0])

        tester.evaluate_J_score()
        scorings.append(tester.J_score)

    print scorings
    print
    print min(scorings), max(scorings)

    bin_a = 0
    bin_b = 12

    incidences = scorings
    bins_edges = generate_linear_range(
        start=bin_a,
        end=bin_b,
        num_points=12)

    print bins_edges
    print incidences

    num_samples = len(incidences)

    title_J_histogram = (
        str(num_samples)
        + " "
        + "sample(s)"
        + " "
        + r"$\theta_{0}\in$"
        + "[" + str(min_th0)
        + "," + str(max_th0)
        + "]")

    fig = matplot.figure()
    axs = matplot.subplot(111)

    fig.suptitle(title_J_histogram)
    axs.set_ylabel("frequency")
    axs.set_xlabel(r"$J$")
    axs.hist(incidences, bins=bins_edges)

    matplot.savefig(exportfiles["savepath1"])
    matplot.close()

    return

def plot_J_score_variabilities_over_th1():

    models = StochyPerc()
    tester = ModelTester()

    exportfiles = write_input_and_save_filenames()
    testing_set = read_and_sort_file_into_datasets(
        exportfiles["largedataset"])
    training_set = read_and_sort_file_into_datasets(
        exportfiles["smalldataset"])

    testset_x = {}

    training_labels = training_set[0]

    model_x0 = numpy.ones(numpy.shape(training_labels))
    tester_x0 = numpy.ones(numpy.shape(testing_set[0]))

    models.prompt_input(
        labels=training_labels,
        feature1=training_set[1],
        feature2=training_set[2],
        feature0=model_x0)

    training_rate = 0.5; upperbound_epoches = 3000
    min_th1 = 1000; max_th1 = 2000; num_models = 1000

    rand_range = generate_range_rand_ints(
        min_th1, max_th1, num_models)

    init_thetas = []; fin_thetas = []

    linrange = generate_linear_range(-2000, 2000, num_models)

    success_ctr = Iterator()

    for i in linrange:
        models.prompt_parameters(
            theta0=1, theta1=i, theta2=1,
            num_epochs=upperbound_epoches,
            training_rate=training_rate)

        models.optimize_params()

        if models.J_score == 0:
            success_ctr.increment()
            fin_thetas.append(models.pile_thetas[-1])
            print success_ctr.read()
        else:
            raise Exception("model not convergent")

    print init_thetas

    scorings = []

    for j in range(len(fin_thetas)):
        optimized_params = fin_thetas[j]

        tester.prompt_model(
            theta0=optimized_params[0],
            theta1=optimized_params[1],
            theta2=optimized_params[2])

        tester.prompt_testset(
            x0=tester_x0,
            x1=testing_set[1],
            x2=testing_set[2],
            labels=testing_set[0])

        tester.evaluate_J_score()

        if tester.J_score == 0:
            raise Exception("perfect score")
        else:
            pass

        scorings.append(tester.J_score)

    print scorings
    print
    print min(scorings), max(scorings)

    bin_a = 0
    bin_b = 12

    incidences = scorings
    bins_edges = generate_linear_range(
        start=bin_a,
        end=bin_b,
        num_points=12)

    print bins_edges
    print incidences

    num_samples = len(incidences)

    title_J_histogram = (
        str(num_samples)
        + " "
        + "sample(s)"
        + " "
        + r"$\theta_{1}\in$"
        + "[" + str(min_th1)
        + "," + str(max_th1)
        + "]")

    fig = matplot.figure()
    axs = matplot.subplot(111)

    fig.suptitle(title_J_histogram)
    axs.set_ylabel("frequency")
    axs.set_xlabel(r"$J$")
    axs.hist(incidences, bins=bins_edges)

    matplot.savefig(exportfiles["savepath1"])
    matplot.close()

    return

plot_J_score_variabilities_over_th0()






