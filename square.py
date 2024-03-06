from typing import Dict, List, Union
from scipy.stats import skew, kurtosis
from math import cos, sin, pi, e
import matplotlib.pyplot as plt
import numpy as np
from particle import decay
import pandas as pd

# Define a square class corresponding to the "Physical Grid"
class Grid:
    # Create the square grid
    def __init__(self, n: int, radius: float):
        self.radius = radius
        self.points: List[Dict[str, float]] = Grid.starting_points(n)
        self.anchors = self._get_anchors()
        self.scorer = decay(self.radius)

    # Function to create a square with n * n points centered around 0
    # Input: INTEGER with dimensions of the square
    # Output: ARRAY with coordinates of all points belonging the square
    @staticmethod
    def starting_points(n: int, include_intermediate: bool = False) -> List[Dict[str, float]]: 
        # Array to store all points
        all_points: List[Dict[str, float]] = []
        # Go through all X-axis values
        for x in range(-n//2+1, n//2+1, 1):
            # Go through all Y-axis values
            for y in range(-n//2+1, n//2+1, 1):
                # Save each point with X, Y points
                coords = {'x': x, 'y': y}
                all_points.append(coords)

                if include_intermediate and y > -n // 2 + 1 and x < n // 2:
                    int_y = y - 0.5
                    int_coords = {'x': x + 0.5, 'y': int_y}
                    all_points.append(int_coords)

        return all_points
    
    def get_coordinates(self) -> List[Dict[str, float]]:
        return self.points
    
    def _get_anchors(self) -> List[Dict[str, float]]:
        min_y = min([point['y'] for point in self.points])
        max_y = max([point['y'] for point in self.points])

        min_x = min([point['x'] for point in self.points])
        max_x = max([point['x'] for point in self.points])

        anchor_one = {'x': min_x, 'y': max_y}
        anchor_two = {'x': max_x, 'y': min_y}

        return [anchor_one, anchor_two]
    
    def _get_anchor_line(self) -> float:
        x1, y1 = self.anchors[0]["x"], self.anchors[0]["y"]
        x2, y2 = self.anchors[1]["x"], self.anchors[1]["y"]

        # Calculate slope and y-intercept
        slope = (y2 - y1) / (x2 - x1)

        return slope

    
    def _get_ray_grid(self, n: int) -> Dict[str, Union[float, np.ndarray, Dict[str, float]]]:
        min_y_grid = min([point['y'] for point in self.points])
        max_y_grid = max([point['y'] for point in self.points])

        min_x_grid = min([point['x'] for point in self.points])
        max_x_grid = max([point['x'] for point in self.points])

        y_grid = np.linspace(min_y_grid-self.radius, max_y_grid+self.radius, endpoint=True, num=n)
        x_grid = np.linspace(min_x_grid-self.radius, max_x_grid+self.radius, endpoint=True, num=n)

        full_grid = []
        for x_coord in x_grid:
            for y_coord in y_grid:
                full_grid.append({
                    'x': x_coord,
                    'y': y_coord
                })

        return {
            'min_grid': min_y_grid,
            'max_grid': max_y_grid,
            'ray_grid': y_grid,
            'full_grid': full_grid,
        }

    def plot_coordinates(self, grid_size: int) -> None:
        (min_y_grid, max_y_grid, ray_grid, _) = self._get_ray_grid(grid_size).values()

        _, ax = plt.subplots()

        for ray in ray_grid:
            plt.axhline(y=ray, color='blue', alpha=0.05)

        for idx, element in enumerate(self.points):
            (x, y) = element.values()

            if idx % 2 == 0:
                color = 'red'
            else:
                color = 'blue'

            # ax.plot(x, y, 'o', color='red')
            circle = plt.Circle((x, y), radius=self.radius, linewidth=1, fill=True, color=color)

            ax.add_patch(circle)

        ax.set_xlim(min_y_grid-0.5, max_y_grid+0.5)
        ax.set_ylim(min_y_grid-0.5, max_y_grid+0.5)
        ax.set_aspect('equal', adjustable='box')
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
            new_x = cos(radian) * point['x'] - sin(radian) * point['y']
            # Compute the new Y-axis value
            new_y = sin(radian) * point['x'] + cos(radian) * point['y']

            # Update the points of the square
            self.points[i] = {'x': new_x, 'y': new_y}

        # Rotate the anchors by applying trigonometrical transformation to all the points
        for i in range(len(self.anchors)):
            # Grab the original point
            point = self.anchors[i]
            # Compute the new X-axis value
            new_x = cos(radian) * point['x'] - sin(radian) * point['y']
            # Compute the new Y-axis value
            new_y = sin(radian) * point['x'] + cos(radian) * point['y']

            # Update the points of anchors
            self.anchors[i] = {'x': new_x, 'y': new_y}

    def _optimize_grid(self, grid: List[Dict[str, float]]):
        slope = self._get_anchor_line()

        points = [point for point in self.points if point['y'] >= point['x'] * slope - 0.1]
        new_grid = [point for point in grid if point['y'] >= point['x'] * slope]

        return {
            'points': points,
            'grid': new_grid,
        }


    # Method to score the square grid its given rotation at the moment
    # Input: ARRAY with the Y coordinates of all fluid rays
    #        FLOAT indicating the radius of action of each point of the square grid (>0.5 and there is overlap)
    # Output: DICTIONARY with the following keys
    #           - mean: mean score for the current grid
    #           - median: median score for the current grid
    #           - stdev: standard deviation of the scores for the current grid
    #           - zeros: amount of rays that do not interact with any points of the grid
    def score_grid(self, grid_size: int) -> Dict[str, float]:
        (_, _, _, full_grid) = self._get_ray_grid(grid_size).values()
        (points, grid) = self._optimize_grid(full_grid).values() 

        # List to store scores 
        grid_scores: List[float] = []

        # Counter for the rays without interactions
        zero_counter = 0
        
        full_progress = len(points) * len(grid)
        curr_progress = 0
        
        # Go through all the rays (fluid)
        for dot in grid:
            
            (x_dot, y_dot) = dot.values()

            # Go through all points in the square grid
            for point in points:
                # Get the y coord of that point
                (x, y) = point.values()

                # Euclidean distance
                distance = np.sqrt((x_dot - x) ** 2 + (y_dot - y) ** 2)

                # Scoring using decay
                point_score = self.scorer(distance)

                # Add score
                grid_scores.append(distance)
                # grid_scores.append(point_score)

                # Progress tracking
                curr_progress += 1 / full_progress

                # If the total score is 0, increase the counter of zeros
                if point_score == 0:
                    zero_counter += 1

        plt.figure()
        plt.hist(grid_scores)

        plt.figure()
        plt.hist(grid_scores)

        plt.show()

        # Compute various statistics taking into account all rays
        answer = {
            'mean': np.mean(grid_scores),
            'median': np.median(grid_scores),
            'stdev': np.std(grid_scores),
            'zeros': zero_counter,
            # 'scores': grid_scores,
        }
       

        # Return the computed statistics
        return answer


my_grid = Grid(n=7, radius=0.49)


all_answers = []

my_grid.rotate_grid(30)
answer = my_grid.score_grid(100)
answer['angle'] = 0
all_answers.append(answer)
print(answer)

# my_grid.plot_coordinates(100)

"""
for i in range(1, 46, 1):
    my_grid.rotate_grid(1)
    answer = my_grid.score_grid(100)


    answer['angle'] = i

    all_answers.append(answer)
"""

    
df = pd.DataFrame(all_answers)
print(df)





"""
# Array to store the best angle for each iteration
all_best_angles = []

# Array with all possible radius of the points of the square grid
# Currently, it goes from 0.50 to 0.01
all_point_radius = [x / 100 for x in list(range(50, 35, -2))]
# Iterate through all possible radius
for idx, point_radius in enumerate(all_point_radius):

    print(f'Loading {idx / len(all_point_radius) * 100}%')

    # Create a square grid with 7x7 points
    my_square = Square(7)

    # Define the initial rotational angle (0 degrees, not radians)
    initial_angle = 0
    # Define the final rotational angle (46 degrees, not radians)
    # Any more than that and the results become redundant
    final_angle = 46
    # How much the angle will increase with each rotation
    step = 1

    # Create array with all rotations to be performed (first rotation is 0º)
    all_angles = [0] + [step for _ in range(initial_angle, final_angle // step + 1, 1)]

    # Define arrays to store the different statistics being tracked
    all_means = []
    all_medians = []
    all_devs = []
    all_zeros = []
    all_sums = []

    # Iterate through all the angles for a given point radius
    for angle in all_angles:

        # Rotate the square grid
        my_square.rotate_square(angle)

        # Compute the height of the square grid
        max_y = max([point['y'] for point in my_square.points])
        min_y = min([point['y'] for point in my_square.points])

        # Create a grid of rays considering the height of the square grid
        # Rays are uniformly distributed across the whole grid
        ray_full_grid = np.linspace(min_y-point_radius, max_y+point_radius, endpoint=True, num=2000)

        # Score the current ray grid (remove the rays on the edge)
        grid_data = my_square.score_grid(ray_full_grid[1:-1], point_radius)

        # Save the results for the current analysis
        all_zeros.append(grid_data['zeros'])
        all_means.append(grid_data['mean'])
        all_medians.append(grid_data['median'])
        all_devs.append(grid_data['stdev'])
        all_sums.append(np.sum(grid_data['scores']))

    # Create an array with all the angles
    x_values = range(initial_angle, final_angle + 2, step)

    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.suptitle(f'Score metrics per rotation angle (radius={point_radius})')


    ax[0].plot(x_values, all_means, '-o', label='mean')
    ax[0].legend()
    ax[1].plot(x_values, all_medians, '-o', label='median')
    ax[1].legend()
    # ax[2].plot(x_values, all_zeros, '-o', label='zeros')
    # ax[2].legend()


    ax[2].plot(x_values, all_devs, '-o', label='deviation')
    ax[2].legend()

    plt.show()
	
    # After computing the metrics for all angles for a given point radius,
    # normalize the results for the different metrics between 0 and 1 for comparability
    norm_mean = [(x-min(all_means)) / (max(all_means) - min(all_means)) for x in all_means]
    norm_median = [(x-min(all_medians)) / (max(all_medians) - min(all_medians)) for x in all_medians]
    norm_dev = [1 - (x-min(all_devs)) / (max(all_devs) - min(all_devs)) for x in all_devs]

    # Compute the final score by combining the 3 metrics mean, median and standard deviation
    # Standard deviation is the most important variable because of consistency
    final_combo_scores = [a * 0 + b * 1 + c * 0 for a, b, c in zip(norm_mean, norm_median, norm_dev)]

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.suptitle(f'Radius: {point_radius}')
    
    ax[0].plot(x_values, final_combo_scores, '-o', label='combo')
    ax[0].legend()
    ax[1].plot(x_values, all_zeros, '-o', label='zeros')
    ax[1].legend()
    
    plt.show()

    # Save the angle for which the highest final score occurs
    all_best_angles.append(x_values[np.argmax(final_combo_scores)])
    # print(x_values[np.argmax(final_combo_scores)])


plt.figure()
plt.title('Best scoring angle for each point radius')
plt.plot(all_point_radius, all_best_angles, '-o')
plt.xticks(all_point_radius)
plt.ylabel('Degrees')
plt.xlabel('Point Radius')

plt.show()
"""