import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class NiftyConfusionMatrix():
    def __init__(self, cm, classes):
        self.cm = cm
        self.classes = classes

    def display(self, title='Confusion Matrix'):
        num_classes = len(self.cm)

        # Create a figure with a white background
        plt.figure(figsize=(10, 9), facecolor='white')
        ax = plt.gca()

        # Get unique colors for each class
        colors = plt.cm.tab20c(np.linspace(0, 1, num_classes))

        # Apply column colors
        for col in range(num_classes):
            # Create a semi-transparent rectangle
            rect = plt.Rectangle((col - 0.5, -0.5), 1, num_classes, color=colors[col], alpha=0.2, zorder=-1)
            ax.add_patch(rect)

        # Apply row colors
        for row in range(num_classes):
            rect = plt.Rectangle((-0.5, row - 0.5), num_classes, 1, color=colors[row], alpha=0.2, zorder=-1)
            ax.add_patch(rect)

        # get max non-diagonal value
        max_value_non_diag = 0
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    if self.cm[i, j] > max_value_non_diag:
                        max_value_non_diag = self.cm[i, j]

        # create color map using max_value_non_diag
        highlight_colors = plt.cm.autumn_r(np.linspace(0, 1, max_value_non_diag + 1))

        # add text and highlight non-zero values
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, self.cm[i, j], ha="center", va="center", color="black", fontsize=9)
                # make non-zero values more visible (but not diagonals)
                if self.cm[i, j] > 0 and i != j:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=highlight_colors[self.cm[i, j]], alpha=0.6, zorder=-1)
                    ax.add_patch(rect)            

        # color diagonal cells unique color
        for i in range(num_classes):
            rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color='gray', alpha=0.3, zorder=-1)
            ax.add_patch(rect)

        # Setup the axes
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(self.classes)
        ax.set_yticklabels(self.classes)

        # Rotate the x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Set limits to ensure all cells are visible
        ax.set_xlim(-0.5, num_classes - 0.5)
        ax.set_ylim(num_classes - 0.5, -0.5)  # Invert y-axis to match conventional matrix display

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        plt.tight_layout()

    def to_csv(self):
        # output confusion matrix to CSV
        cm_df = pd.DataFrame(self.cm, index=self.classes, columns=self.classes)
        cm_df.to_csv("confusion_matrix.csv")
