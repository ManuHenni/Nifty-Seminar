import numpy as np
class Line:
    def __init__(self, vector, point_on_line) -> None:
        self.vector = np.array(vector).reshape(3, 1)
        self.point_on_line = np.array(point_on_line).reshape(3, 1)

    def get_plot_points(self, step_size=0.01):
        points = np.zeros((3, 0))
        for i in np.arange(0, 1, step_size):
            points = np.concatenate((points, self.point_on_line + i * self.vector), axis=1)
        return points
