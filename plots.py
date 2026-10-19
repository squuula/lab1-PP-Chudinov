"""Гистограммы значений случайной матрицы по строкам и столбцам (часть 2, задача 4).

Запуск: python plots.py
"""

import matplotlib.pyplot as plt

from grader_contracts.numpy_tasks import RandomMatrixInput
from numpy_tasks import matrix_statistics, plot_matrix_histograms


def main():
    stats = matrix_statistics(RandomMatrixInput(rows=5, columns=6, mean=0.0, std=1.0, seed=42))
    print('Средние по строкам:', stats.row_means)
    print('Средние по столбцам:', stats.column_means)
    print('Дисперсии по строкам:', stats.row_variances)
    print('Дисперсии по столбцам:', stats.column_variances)

    fig = plot_matrix_histograms(stats, bins=8, show=False)
    fig.savefig('histograms.png', dpi=120)
    plt.show()


if __name__ == '__main__':
    main()
