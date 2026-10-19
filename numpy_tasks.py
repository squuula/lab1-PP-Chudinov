"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    if matrices.ndim == 2:          # одна матрица без размерности пакета
        matrices = matrices[np.newaxis]
    p, n, _ = matrices.shape
    vectors = vectors.reshape(p, n, 1)
    # matrices @ vectors даёт p векторов (n, 1), складываем их по оси пакета
    return (matrices @ vectors).sum(axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    matrix = np.asarray(matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    matrix = np.asarray(matrix)
    return [np.unique(column).tolist() for column in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def plot_matrix_histograms(stats: MatrixStatistics, bins: int = 10, show: bool = True):
    """Гистограммы значений по строкам и по столбцам (matplotlib.pyplot.hist)."""
    import matplotlib.pyplot as plt

    matrix = np.asarray(stats.matrix)
    fig, (ax_rows, ax_cols) = plt.subplots(1, 2, figsize=(12, 4))

    for i, row in enumerate(matrix):
        ax_rows.hist(row, bins=bins, alpha=0.5, label=f'строка {i}')
    ax_rows.set_title('Значения по строкам')
    ax_rows.set_xlabel('значение')
    ax_rows.set_ylabel('частота')

    for j, column in enumerate(matrix.T):
        ax_cols.hist(column, bins=bins, alpha=0.5, label=f'столбец {j}')
    ax_cols.set_title('Значения по столбцам')
    ax_cols.set_xlabel('значение')
    ax_cols.set_ylabel('частота')

    if matrix.shape[0] <= 10:
        ax_rows.legend(fontsize='small')
    if matrix.shape[1] <= 10:
        ax_cols.legend(fontsize='small')

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    row_idx, col_idx = np.indices((rows, columns))
    return np.where((row_idx + col_idx) % 2 == 0, first, second)


def _blank_image(height: int, width: int, color) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = _blank_image(image_height, image_width, background_color)

    top = (image_height - height) // 2
    left = (image_width - width) // 2
    # если прямоугольник больше картинки, обрезаем по краям
    y_start, y_end = max(top, 0), min(top + height, image_height)
    x_start, x_end = max(left, 0), min(left + width, image_width)
    image[y_start:y_end, x_start:x_end] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = _blank_image(image_height, image_width, background_color)
    if semi_axis_x <= 0 or semi_axis_y <= 0:
        return image

    x0 = (image_width - 1) / 2
    y0 = (image_height - 1) / 2
    y, x = np.ogrid[:image_height, :image_width]
    mask = ((x - x0) ** 2 / semi_axis_x ** 2 + (y - y0) ** 2 / semi_axis_y ** 2) <= 1
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        raise ValueError('Ряд не должен быть пустым')
    if not 1 <= window <= n:
        raise ValueError('Размер окна должен быть от 1 до длины ряда')

    if n >= 3:
        middle = values[1:-1]
        maxima = np.where((middle > values[:-2]) & (middle > values[2:]))[0] + 1
        minima = np.where((middle < values[:-2]) & (middle < values[2:]))[0] + 1
    else:
        maxima = minima = np.array([], dtype=int)

    moving_average = np.convolve(values, np.ones(window) / window, mode='valid')

    return TimeSeriesStatistics(
        mean=float(values.mean()),
        variance=float(values.var()),
        std=float(values.std()),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int).ravel()
    if labels.size and labels.min() < 0:
        raise ValueError('Метки должны быть неотрицательными')
    needed = int(labels.max()) + 1 if labels.size else 0
    if class_count is None:
        class_count = needed
    elif class_count < needed:
        raise ValueError('class_count меньше, чем максимальная метка + 1')

    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result
