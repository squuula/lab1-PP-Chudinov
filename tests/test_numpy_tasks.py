import numpy as np
import pytest

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)
from numpy_tasks import (
    analyze_time_series, binarize, chess, draw_ellipse, draw_rectangle,
    matrix_statistics, one_hot, plot_matrix_histograms, sum_prod,
    unique_columns, unique_rows,
)

WHITE, RED = (255, 255, 255), (255, 0, 0)


# Задача 1
def test_sum_prod_identity():
    matrices = [np.eye(2), np.eye(2)]
    vectors = [np.array([[1], [2]]), np.array([[3], [4]])]
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    assert result.shape == (2, 1)
    np.testing.assert_array_equal(result, [[4], [6]])


def test_sum_prod_random_matches_loop():
    rng = np.random.default_rng(0)
    p, n = 5, 4
    matrices = rng.normal(size=(p, n, n))
    vectors = rng.normal(size=(p, n, 1))
    expected = sum(matrices[i] @ vectors[i] for i in range(p))
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    assert result.shape == (n, 1)
    np.testing.assert_allclose(result, expected)


def test_sum_prod_single():
    result = sum_prod(MatrixVectorBatchInput([[[2, 0], [0, 3]]], [[[1], [1]]]))
    np.testing.assert_array_equal(result, [[2], [3]])


# Задача 2
def test_binarize():
    m = np.array([[0.1, 0.5, 0.9], [1.0, -2.0, 0.51]])
    result = binarize(BinarizeInput(m))
    np.testing.assert_array_equal(result, [[0, 0, 1], [1, 0, 1]])
    np.testing.assert_array_equal(m, [[0.1, 0.5, 0.9], [1.0, -2.0, 0.51]])  # исходная не изменена
    assert result is not m


def test_binarize_threshold_equal_goes_to_zero():
    result = binarize(BinarizeInput([[3, 5, 7]], threshold=5))
    np.testing.assert_array_equal(result, [[0, 0, 1]])


# Задача 3
def test_unique_rows_and_columns():
    m = np.array([[1, 2, 2], [3, 3, 3], [1, 2, 3]])
    assert unique_rows(MatrixInput(m)) == [[1, 2], [3], [1, 2, 3]]
    assert unique_columns(MatrixInput(m)) == [[1, 3], [2, 3], [2, 3]]


def test_unique_single_element():
    assert unique_rows(MatrixInput([[5]])) == [[5]]
    assert unique_columns(MatrixInput([[5]])) == [[5]]


# Задача 4
def test_matrix_statistics():
    stats = matrix_statistics(RandomMatrixInput(rows=200, columns=300, mean=5, std=2, seed=1))
    assert isinstance(stats, MatrixStatistics)
    m = stats.matrix
    assert m.shape == (200, 300)
    np.testing.assert_allclose(stats.row_means, m.mean(axis=1))
    np.testing.assert_allclose(stats.column_means, m.mean(axis=0))
    np.testing.assert_allclose(stats.row_variances, m.var(axis=1))
    np.testing.assert_allclose(stats.column_variances, m.var(axis=0))
    assert stats.row_means.shape == (200,) and stats.column_means.shape == (300,)
    assert abs(m.mean() - 5) < 0.05
    assert abs(m.std() - 2) < 0.05


def test_matrix_statistics_seed_reproducible():
    a = matrix_statistics(RandomMatrixInput(3, 4, seed=7)).matrix
    b = matrix_statistics(RandomMatrixInput(3, 4, seed=7)).matrix
    c = matrix_statistics(RandomMatrixInput(3, 4, seed=8)).matrix
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_histograms_build(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    stats = matrix_statistics(RandomMatrixInput(4, 5))
    fig = plot_matrix_histograms(stats, show=False)
    fig.savefig(tmp_path / 'hist.png')
    assert (tmp_path / 'hist.png').exists()


# Задача 5
def test_chess_example():
    result = chess(ChessInput(3, 4, 0, 1))
    np.testing.assert_array_equal(result, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])


def test_chess_edge_cases():
    np.testing.assert_array_equal(chess(ChessInput(1, 1, 7, 9)), [[7]])
    np.testing.assert_array_equal(chess(ChessInput(1, 3, 2, 5)), [[2, 5, 2]])
    r = chess(ChessInput(5, 6, -1.5, 2.5))
    assert r[0, 0] == -1.5
    assert np.all(r[:, 1:] != r[:, :-1]) and np.all(r[1:, :] != r[:-1, :])


# Задача 6
def test_draw_rectangle():
    img = draw_rectangle(RectangleInput(width=4, height=2, image_height=6, image_width=8,
                                        shape_color=RED, background_color=WHITE))
    assert img.shape == (6, 8, 3)
    mask = np.all(img == RED, axis=2)
    expected = np.zeros((6, 8), dtype=bool)
    expected[2:4, 2:6] = True
    np.testing.assert_array_equal(mask, expected)
    assert np.all(img[~mask] == WHITE)


def test_draw_rectangle_bigger_than_image():
    img = draw_rectangle(RectangleInput(20, 20, 5, 5, RED, WHITE))
    assert img.shape == (5, 5, 3)
    assert np.all(img == RED)


def test_draw_ellipse():
    img = draw_ellipse(EllipseInput(semi_axis_x=10, semi_axis_y=5, image_height=21, image_width=31,
                                    shape_color=RED, background_color=WHITE))
    assert img.shape == (21, 31, 3)
    mask = np.all(img == RED, axis=2)
    assert mask[10, 15]                         # центр
    assert mask[10, 5] and mask[10, 25]         # концы горизонтальной полуоси
    assert mask[5, 15] and mask[15, 15]         # концы вертикальной полуоси
    assert not mask[10, 4] and not mask[4, 15]  # сразу за границей
    assert not mask[0, 0]                       # угол — фон
    np.testing.assert_array_equal(mask, mask[::-1, ::-1])  # симметрия


def test_draw_circle_area():
    img = draw_ellipse(EllipseInput(50, 50, 201, 201, RED, WHITE))
    area = np.all(img == RED, axis=2).sum()
    assert abs(area - np.pi * 50 ** 2) / (np.pi * 50 ** 2) < 0.02


# Задача 7
def test_analyze_time_series():
    values = [1, 3, 2, 5, 4, 4, 1, 2]
    stats = analyze_time_series(TimeSeriesInput(values, window=3))
    assert isinstance(stats, TimeSeriesStatistics)
    arr = np.array(values, dtype=float)
    assert stats.mean == pytest.approx(arr.mean())
    assert stats.variance == pytest.approx(arr.var())
    assert stats.std == pytest.approx(arr.std())
    # 4, 4 — плато, строгим максимумом не является
    np.testing.assert_array_equal(stats.local_maxima_indices, [1, 3])
    np.testing.assert_array_equal(stats.local_minima_indices, [2, 6])
    expected_ma = [np.mean(arr[i:i + 3]) for i in range(len(arr) - 2)]
    np.testing.assert_allclose(stats.moving_average, expected_ma)
    assert len(stats.moving_average) == len(values) - 3 + 1


def test_time_series_edge_cases():
    s = analyze_time_series(TimeSeriesInput([5, 5], window=2))
    assert len(s.local_maxima_indices) == 0 and len(s.local_minima_indices) == 0
    np.testing.assert_allclose(s.moving_average, [5.0])
    s = analyze_time_series(TimeSeriesInput([1, 2, 3], window=1))
    np.testing.assert_allclose(s.moving_average, [1, 2, 3])
    with pytest.raises(ValueError):
        analyze_time_series(TimeSeriesInput([1, 2], window=3))


# Задача 8
def test_one_hot_example():
    result = one_hot(OneHotInput([0, 2, 3, 0]))
    np.testing.assert_array_equal(result, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])


def test_one_hot_class_count():
    result = one_hot(OneHotInput([1, 0], class_count=5))
    assert result.shape == (2, 5)
    np.testing.assert_array_equal(result, [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
    with pytest.raises(ValueError):
        one_hot(OneHotInput([4], class_count=3))


def test_one_hot_single_zero():
    np.testing.assert_array_equal(one_hot(OneHotInput([0])), [[1]])
