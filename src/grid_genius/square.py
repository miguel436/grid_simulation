from math import cos, pi, sin
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from grid_genius.models import GridScore, Point, RayGrid
from grid_genius.particle import decay


# Define a square class corresponding to the "Physical Grid"
class Grid:
    # Create the square grid
    def __init__(self, n: int, radius: float):
        self.radius = radius
        self.points: List[Point] = [
            Point(**p) for p in Grid.starting_points(n)
        ]
        self.anchors = self._get_anchors()
        self.scorer = decay(self.radius + 0.04)

    # Function to create a square with n * n points centered around 0
    # Input: INTEGER with dimensions of the square
    # Output: ARRAY with coordinates of all points belonging the square
    @staticmethod
    def starting_points(
        n: int, include_intermediate: bool = False
    ) -> List[Dict[str, float]]:
        # Array to store all points
        all_points: List[Dict[str, float]] = []
        # Go through all X-axis values
        for x in range(-n // 2 + 1, n // 2 + 1, 1):
            # Go through all Y-axis values
            for y in range(-n // 2 + 1, n // 2 + 1, 1):
                # Save each point with X, Y points
                coords = {"x": x, "y": y}
                all_points.append(coords)

                if include_intermediate and y > -n // 2 + 1 and x < n // 2:
                    int_y = y - 0.5
                    int_coords = {"x": x + 0.5, "y": int_y}
                    all_points.append(int_coords)

        return all_points

    def _get_anchors(self) -> List[Dict[str, float]]:
        min_y = min([point.y for point in self.points])
        max_y = max([point.y for point in self.points])
        min_x = min([point.x for point in self.points])
        max_x = max([point.x for point in self.points])

        # Convert to Point objects instead of dictionaries
        anchor_one = Point(x=min_x, y=max_y)
        anchor_two = Point(x=max_x, y=min_y)

        return [anchor_one, anchor_two]

    def _get_ray_grid(self, step: Optional[float] = 0.0075) -> RayGrid:
        min_y_grid = min([point.y for point in self.points])
        max_y_grid = max([point.y for point in self.points])

        y_grid = np.arange(
            min_y_grid - self.radius,
            max_y_grid + self.radius,
            step=float(step),
        )

        return RayGrid(
            min_grid=min_y_grid, max_grid=max_y_grid, ray_grid=y_grid
        )

    def _get_horizontal_stream(self) -> List[float]:
        min_x_grid = min([point.x for point in self.points])
        max_x_grid = max([point.x for point in self.points])

        x_grid = np.arange(
            min_x_grid - self.radius, max_x_grid + self.radius, step=0.0075
        )

        return x_grid

    def plot_coordinates(
        self,
        step: Optional[float] = None,
        color_scheme: None | List[str] = None,
        gray_scale: bool = False,
    ) -> None:
        ray_grid_data = (
            self._get_ray_grid(step) if step else self._get_ray_grid()
        )
        min_y_grid, max_y_grid, ray_grid = ray_grid_data.model_dump().values()

        if color_scheme is None:
            color_scheme = ["black" for _ in ray_grid]

        _, ax = plt.subplots()
        for ray, color in zip(ray_grid, color_scheme):
            if color != "":
                plt.axhline(y=ray, color=color)

        for idx, element in enumerate(self.points):
            (x, y) = element.model_dump().values()

            if gray_scale:
                color = "lightgray"
            else:
                if idx % 2 == 0:
                    color = "red"
                else:
                    color = "blue"

            # ax.plot(x, y, 'o', color='red')
            circle = plt.Circle(
                (x, y), radius=self.radius, linewidth=1, fill=True, color=color
            )

            ax.add_patch(circle)

        ax.set_xlim(min_y_grid - 0.5, max_y_grid + 0.5)
        ax.set_ylim(min_y_grid - 0.5, max_y_grid + 0.5)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title("Tilted Grid Example")
        ax.set_ylabel("Arbitrary Units")
        ax.set_xlabel("Arbitrary Units")
        plt.axis("off")
        plt.show()

    # Method to rotate the square grid around the point X = 0; Y = 0
    # Input: INTEGER of rotation angle in degrees
    def rotate_grid(self, angle: float) -> None:
        # Convert degrees to radians
        radian = pi / 180 * angle

        # Math reference to perform rotations
        # In our case (x0, y0) = (0, 0)
        # cos(theta) * (px-x0) - sin(theta) * (py-y0) + x0
        # sin(theta) * (px - x0) + cos(theta) * (py - y0) + y0

        # Rotate the square grid by applying trigonometrical transformation to all the points
        for i in range(len(self.points)):
            # Grab the original point
            point = self.points[i]
            # Compute the new X-axis value
            new_x = cos(radian) * point.x - sin(radian) * point.y
            # Compute the new Y-axis value
            new_y = sin(radian) * point.x + cos(radian) * point.y

            # Update the points of the square
            self.points[i] = Point(x=new_x, y=new_y)

        # Rotate the anchors by applying trigonometrical transformation to all the points
        for i in range(len(self.anchors)):
            # Grab the original point
            point = self.anchors[i]
            # Compute the new X-axis value
            new_x = cos(radian) * point.x - sin(radian) * point.y
            # Compute the new Y-axis value
            new_y = sin(radian) * point.x + cos(radian) * point.y

            # Update the points of anchors
            self.anchors[i] = Point(x=new_x, y=new_y)

    # Method to score the square grid its given rotation at the moment
    # Input: ARRAY with the Y coordinates of all fluid rays
    #        FLOAT indicating the radius of action of each point of the square grid (>0.5 and there is overlap)
    # Output: DICTIONARY with the following keys
    #           - mean: mean score for the current grid
    #           - median: median score for the current grid
    #           - stdev: standard deviation of the scores for the current grid
    #           - zeros: amount of rays that do not interact with any points of the grid
    def score_grid(self) -> GridScore:
        (_, _, ray_grid) = self._get_ray_grid().model_dump().values()

        # Counter for the rays without interactions
        zero_counter = 0
        all_ray_scores = []

        # Go through all the rays (fluid)
        for ray in ray_grid:

            y_ray = ray
            x_rays = self._get_horizontal_stream()

            ray_score = 0

            for x_ray in x_rays:
                # Go through all points in the square grid
                for point in self.points:
                    # Get the y coord of that point
                    (x, y) = point.model_dump().values()

                    # Euclidean distance
                    distance = np.sqrt((x_ray - x) ** 2 + (y_ray - y) ** 2)

                    # Scoring using decay
                    point_score = self.scorer(distance)

                    # Add score
                    ray_score += point_score

            # If the total score is 0, increase the counter of zeros
            if ray_score == 0:
                zero_counter += 1

            all_ray_scores.append(ray_score)

        # Return the computed statistics
        return GridScore(
            mean=np.mean(all_ray_scores),
            median=np.median(all_ray_scores),
            stdev=np.std(all_ray_scores),
            zeros=zero_counter,
            scores=all_ray_scores,
        )
