import torch
import torch.nn as nn
import csv


class MultilayerPerceptronClassifier(nn.Module):

    def __init__(self, num_features, hidden_dims, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(num_features, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.linear3 = nn.Linear(hidden_dims, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            prediction = self.forward(x)
            prediction = torch.softmax(prediction, dim=1)
            prediction = torch.argmax(prediction, dim=1)
        return prediction

    def learn(self, train_dataloader, dev_dataloader, num_epochs, optimizer, loss_fct):

        for epoch in range(num_epochs):
            print("Training Epoch " + str(epoch + 1) + "...")

            total_train_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                predictions = self.forward(batch_X)
                loss = loss_fct(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * batch_X.size(0)

                _, predicted_labels = torch.max(
                    predictions, 1
                )  # Returns maximum value and corresponding index
                _, actual_labels = torch.max(batch_y, 1)
                correct_predictions += (predicted_labels == actual_labels).sum().item()
                total_samples += batch_y.size(0)
            train_loss = total_train_loss / len(train_dataloader.dataset)
            train_accuracy = correct_predictions / total_samples

            total_dev_loss = 0.0
            correct_dev_predictions = 0
            total_dev_samples = 0
            for batch_X, batch_y in dev_dataloader:
                with torch.no_grad():
                    predictions = self.forward(batch_X)
                loss = loss_fct(predictions, batch_y)
                total_dev_loss += loss.item() * batch_X.size(0)

                _, predicted_dev_labels = torch.max(predictions, 1)
                _, actual_labels = torch.max(batch_y, 1)
                correct_dev_predictions += (
                    (predicted_dev_labels == actual_labels).sum().item()
                )
                total_dev_samples += batch_y.size(0)

            dev_loss = total_dev_loss / len(dev_dataloader.dataset)
            dev_accuracy = correct_dev_predictions / total_dev_samples

            print(
                "Train Loss: "
                + str(train_loss)
                + " | Train Accuracy: "
                + str(int(train_accuracy * 100))
                + " | Dev Loss: "
                + str(dev_loss)
                + " | Dev Accuracy: "
                + str(int(dev_accuracy * 100))
            )

    def evaluate(self, test_dataloader, loss_fct, output_csv=None):

        print("Evaluating Model On Test Set")

        total_test_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch_X, batch_y in test_dataloader:
            predictions = self.predict(batch_X)
            loss = loss_fct(predictions, batch_y)
            total_test_loss += loss.item() * batch_X.size(0)

            all_predictions.extend(predictions.tolist())
            all_labels.extend(batch_y.tolist())
        test_loss = total_test_loss / len(test_dataloader.dataset)

        print("Test Loss: " + str(test_loss))

        if output_csv:
            with open(output_csv, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Prediction", "Actual"])
                for pred, actual in zip(all_predictions, all_labels):
                    writer.writerow([pred[0], actual])
