import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    """
    Plot learning curve with running average

    Args:
        x: Episode numbers
        scores: Scores for each episode
        figure_file: Path to save the figure
    """
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])

    plt.figure(figsize=(10, 6))
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(figure_file, dpi=150, bbox_inches='tight')
    plt.close()