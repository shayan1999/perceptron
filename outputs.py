import matplotlib.pyplot as plt
from variables import colors, save_address, plot_size
import pandas as pd;

def actual_data_plot(scaled_data, labels, name, title):
    plt.figure(figsize=(plot_size['x'], plot_size['y']))
    plt.scatter(scaled_data[labels == 0][:, 0], scaled_data[labels == 0][:, 1], c=colors['label_zero'], label='class: 0')
    plt.scatter(scaled_data[labels == 1][:, 0], scaled_data[labels == 1][:, 1], c=colors['label_one'], label='class: 1')
    plt.xlabel('x1 (normalized)')
    plt.ylabel('x2 (normalized)')
    plt.title(f"{title}")
    plt.legend()
    plt.savefig(f"{save_address}/{name}")
    plt.clf()

def decision_boundary_plot(x_meshgrid, y_meshgrid, meshgrid_predicts, scaled_data, labels):
    plt.figure(figsize=(plot_size['x'], plot_size['y']))
    plt.contourf(x_meshgrid, y_meshgrid, meshgrid_predicts, alpha=0.3)
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels, cmap='viridis', edgecolors='k', s=20)
    plt.xlabel('x1 (normalized)')
    plt.ylabel('x2 (normalized)')
    plt.title('Perceptron Decision Boundaries')
    plt.savefig(f"{save_address}/decision_boundary_plot")
    plt.clf()

def plot_wrong_predictions_poly(wrong_idx, t_data_scaled, t_labels, v_data_scaled, v_labels, degree):
    plt.figure(figsize=(plot_size['x'], plot_size['y']))
    plt.scatter(t_data_scaled[:, 0], t_data_scaled[:, 1], c=t_labels, cmap='viridis', edgecolors='k', s=20)
    plt.scatter(v_data_scaled[:, 0], v_data_scaled[:, 1], c=v_labels, cmap='viridis', edgecolors='k', s=20, marker='D')
    plt.scatter(v_data_scaled[wrong_idx, 0], v_data_scaled[wrong_idx, 1], c='red', edgecolors='k', s=100, marker='X', label='Wrong Validation Predictions')
    plt.scatter(t_data_scaled[wrong_idx, 0], t_data_scaled[wrong_idx, 1], c='blue', edgecolors='k', s=100, marker='X', label='Wrong Train Predictions')
    plt.xlabel('x1 (normalized)')
    plt.ylabel('x2 (normalized)')
    plt.title('Wrong Predictions Highlighted')
    plt.legend()
    plt.savefig(f"{save_address}/wrong_predicts_degree_{degree}")
    plt.clf()


def result_maker(train_accuracy, val_accuracy, results, test_predictions, weights, bias):
    pd.Series(test_predictions, name='labels').to_csv(f"{save_address}/preds.csv", index=False)

    with open(save_address+'/results.txt', 'w') as file:
        file.write("_____first train accuracy_____\n")
        file.write("Training Accuracy: {:.2f}%\n".format(train_accuracy * 100))
        file.write("Training Accuracy: {:.2f}%\n".format(val_accuracy * 100))
        file.write(f"weights: ${weights}\n")
        file.write(f"bias: ${bias}\n")
        file.write("______________________________\n")
        file.write("__________polynomials_________ \n")
        file.write(f"Degree 2 - Train Accuracy: {results[2]['Train']:.2f}, Validation Accuracy: {results[2]['Validation']:.2f}\n")
        file.write(f"weights: ${results[2]['Weights']}\n")
        file.write(f"bias: ${results[2]['Bias']}\n")
        file.write("______________________________\n")
        file.write(f"Degree 3 - Train Accuracy: {results[3]['Train']:.2f}, Validation Accuracy: {results[3]['Validation']:.2f}\n")
        file.write(f"weights: ${results[3]['Weights']}\n")
        file.write(f"bias: ${results[3]['Bias']}\n")
        file.write("______________________________\n")
        file.write(f"Degree 5 - Train Accuracy: {results[5]['Train']:.2f}, Validation Accuracy: {results[5]['Validation']:.2f}\n")
        file.write(f"weights: ${results[5]['Weights']}\n")
        file.write(f"bias: ${results[5]['Bias']}\n")
        file.write("______________________________\n")
        file.write(f"Degree 10 - Train Accuracy: {results[10]['Train']:.2f}, Validation Accuracy: {results[10]['Validation']:.2f}\n")
        file.write(f"weights: ${results[10]['Weights']}\n")
        file.write(f"bias: ${results[10]['Bias']}\n")

