import click

from grid_genius.square import Grid


@click.command()
@click.option(
    "--angle", type=float, required=True, help="The rotation angle in degrees"
)
def main(angle: float):
    my_grid = Grid(n=7, radius=0.49)
    my_grid.rotate_grid(angle)
    my_grid.plot_coordinates(step=0.075)


if __name__ == "__main__":
    main()
