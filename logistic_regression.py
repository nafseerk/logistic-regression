from data_loader import DataLoader
import numpy as np
import pandas as pd
import pprint


class LogisticRegression:

    def __init__(self, M):
        self.M = M
        self.y_map = {5.0: 1, 6.0: 0}
        self.w_bar = None
        self.max_iterations = 10
        self.train_accuracy = None

    def sigmoid(self, x):
        power = -1 * np.add(np.matmul(np.transpose(self.w_bar[1:]), x), self.w_bar[0, 0])
        return 1 / (1 + np.exp(power))

    def get_gradient_and_hessian_and_log_likelihood(self, dataset):

        gradient = np.zeros((self.M + 1, 1), dtype=float)
        H = np.zeros((self.M + 1, self.M + 1), dtype=float)
        log_likelihood = 0
        for train_set_attrs, train_set_labels in dataset:

            if len(train_set_attrs) != len(train_set_labels):
                raise ValueError('Count mismatch between attributes and labels')

            for i, row in train_set_attrs.iterrows():
                xi = row.values.reshape((self.M, 1))
                xi_bar = np.append(1, xi).reshape((self.M + 1, 1))
                yi = self.y_map[train_set_labels.iat[i, 0]]
                sigmoid_value = self.sigmoid(xi)
                gradient = np.add(gradient, (sigmoid_value - yi) * xi_bar)
                H = np.add(H, sigmoid_value * (1 - sigmoid_value) * np.matmul(xi_bar, np.transpose(xi_bar)))
                log_likelihood += (yi * np.log(sigmoid_value) + (1 - yi) * np.log(1 - sigmoid_value))

        return H, gradient, log_likelihood

    def learn(self, dataset, report_acc=False, max_iterations=10):

        self.max_iterations = max_iterations

        self.w_bar = np.zeros((self.M + 1, 1))
        small_change = 0.0000000001
        log_likelihood_old = self.get_gradient_and_hessian_and_log_likelihood(dataset)[2]
        change_in_log_likelihood = np.Infinity

        i = 0
        while abs(change_in_log_likelihood) > small_change and i < self.max_iterations:
            H, gradient, log_likelihood_new = self.get_gradient_and_hessian_and_log_likelihood(dataset)
            H_inv = np.linalg.inv(H)
            self.w_bar = self.w_bar - np.matmul(H_inv, gradient)
            change_in_log_likelihood = log_likelihood_old - log_likelihood_new
            log_likelihood_old = log_likelihood_new
            i += 1

        if report_acc:
            self.train_accuracy = self.k_fold_cross_validation(dataset)
            print('Training Accuracy = %.3f %%' % self.train_accuracy)

    def classify_point(self, x):
        prob_y_given_x = self.sigmoid(x)[0][0]
        y = 1 if prob_y_given_x >= 0.5 else 0
        for label, yi in self.y_map.items():
            if yi == y:
                return label

    # Returns the value of the equation of the separating hyperplane
    def get_equation_value(self, x):
        x_bar = np.append(1, x)
        val = np.matmul(np.transpose(self.w_bar), x_bar)
        return val

    # For each point in the dataset, check on which side of the hyperplane, the point lies
    def do_experiment(self, test_attrs, true_labels=None):
        N = len(test_attrs)
        if not true_labels.empty:
            if len(test_attrs) != len(true_labels):
                raise ValueError('count mismatch in attributes and labels')

        below = []
        above = []
        for i, row in test_attrs.iterrows():
            xi = row.values.reshape((self.M, 1))
            val = self.get_equation_value(xi)
            if val < 0:
                below.append(true_labels.iat[i, 0])
            else:
                above.append(true_labels.iat[i, 0])

        return below, above

    def classify(self, test_attrs, true_labels=None):
        N = len(test_attrs)
        if not true_labels.empty:
            if len(test_attrs) != len(true_labels):
                raise ValueError('count mismatch in attributes and labels')

        correct = 0
        predicted_labels = []
        for i, row in test_attrs.iterrows():
            xi = row.values.reshape((self.M, 1))
            predicted_label = self.classify_point(xi)
            predicted_labels.append(predicted_label)
            if not true_labels.empty:
                true_label = true_labels.iat[i, 0]
                if predicted_label == true_label:
                    correct += 1

        accuracy = None
        if true_labels is not None:
            accuracy = correct / N * 100

        predicted_labels = pd.DataFrame(np.array(predicted_labels))
        return predicted_labels, accuracy

    def k_fold_cross_validation(self, dataset, k=10):
        cv_test_model = LogisticRegression(M=self.M)
        avg_accuracy = 0.0
        for i in range(k):
            test_attrs, test_labels = dataset.pop(0)
            cv_test_model.learn(dataset)
            accuracy = cv_test_model.classify(test_attrs, true_labels=test_labels)[1]
            dataset.append((test_attrs, test_labels))
            avg_accuracy += accuracy
        self.train_accuracy = avg_accuracy / k
        print('Training Accuracy', self.train_accuracy)
        return self.train_accuracy

    def summary(self):
        print('\n=====Model Summary=====')
        print('w0 =', self.w_bar[0])
        print('\nw of size', end=' ')
        print(self.w_bar[1:].shape, ':')
        pprint.pprint(self.w_bar[1:])
        if self.train_accuracy:
            print('Training Accuracy = %.3f %%' % self.train_accuracy)


if __name__ == '__main__':
    full_dataset = DataLoader.load_full_dataset('./dataset')
    log_reg = LogisticRegression(M=64)
    log_reg.learn(full_dataset)

    log_reg.summary()

    attrs, labels = DataLoader.load_merged_dataset('./dataset')
    below, above = log_reg.do_experiment(attrs, true_labels=labels)
    print('No of points in class 5 below hyperplane= %d' % below.count(5.0))
    print('No of points in class 6 below hyperplane= %d' % below.count(6.0))
    print('No of points in class 5 above hyperplane= %d' % above.count(5.0))
    print('No of points in class 6 above hyperplane= %d' % above.count(6.0))