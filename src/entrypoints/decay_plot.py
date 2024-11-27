import click
import matplotlib.pyplot as plt
import numpy as np

from grid_genius.particle import decay


@click.command()
@click.option("--radius", type=float, required=True, help="Radius of clusters")
def main(radius: float):
    decay_func = decay(radius)

    # Create x values from 0 to slightly above radius
    x_values = np.linspace(0, radius, 200)
    y_values = [decay_func(x) for x in x_values]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, "b-", label="Decay Function")
    plt.axvline(x=radius, color="r", linestyle="--", label="radius")

    # Add labels and title
    plt.xlabel("x")
    plt.ylabel("Score")
    plt.title(f"Decay Function (radius={radius})")
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
