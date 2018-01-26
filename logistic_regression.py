from data_loader import DataLoader
import numpy as np
import pandas as pd


class LogisticRegression:

    def __init__(self, M):
        self.M = M
        self.y_map = {5.0: 1, 6.0: 0}
        self.w_bar = None
        self.max_iterations = 50
        self.train_accuracy = None

    def sigmoid(self, x):
        power = -1 * np.add(np.matmul(np.transpose(self.w_bar[1:]), x), self.w_bar[0, 0])
        return 1 / (1 + np.exp(power))

    def get_gradient_and_hessian(self, dataset):

        gradient = np.zeros((self.M + 1, 1), dtype=float)
        H = np.zeros((self.M + 1, self.M + 1), dtype=float)
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

        return H, gradient

    def learn(self, dataset, verbose=False, report_acc=False):

        # TODO - set initial values for weights
        self.w_bar = np.zeros((self.M + 1, 1))

        # TODO - add logic for convergence check
        for i in range(self.max_iterations):
            H, gradient = self.get_gradient_and_hessian(dataset)
            H_inv = np.linalg.inv(H)
            self.w_bar = self.w_bar - np.matmul(H_inv, gradient)

        if report_acc:
            self.train_accuracy = self.k_fold_cross_validation(full_dataset)
            print('Training Accuracy = %.2f %%' % self.train_accuracy)

    def classify_point(self, x):
        prob_y_given_x = self.sigmoid(x)[0][0]
        y = 1 if prob_y_given_x >= 0.5 else 0
        for label, yi in self.y_map.items():
            if yi == y:
                return label

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
        avg_accuracy = 0.0
        for i in range(k):
            test_attrs, test_labels = dataset.pop(0)
            accuracy = self.classify(test_attrs, true_labels=test_labels)[1]
            dataset.append((test_attrs, test_labels))
            avg_accuracy += accuracy
        return avg_accuracy / k


if __name__ == '__main__':
    full_dataset = DataLoader.load_full_dataset('./dataset')
    log_reg = LogisticRegression(M=64)
    log_reg.learn(full_dataset, report_acc=True)

    # Test the model with test data taken from training data
    train_dataset, test_attrs, test_labels = DataLoader.load_with_test_data(
        './dataset',
        split_ratio=0.1)

    log_reg = LogisticRegression(M=64)
    log_reg.learn(train_dataset)
    predictions, acc = log_reg.classify(test_attrs, true_labels=test_labels)

    print('Test Accuracy = %.2f %%' % acc)
    print('=====Predictions=====')
    print(predictions)