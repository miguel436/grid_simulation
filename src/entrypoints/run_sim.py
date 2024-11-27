import pandas as pd

from grid_genius.square import Grid


def main():
    my_grid = Grid(n=7, radius=0.49)
    print("Simulation started. This is gonna take a while!")

    all_answers = []
    answer = my_grid.score_grid()
    answer.angle = 0
    all_answers.append(answer.model_dump())

    print(0)

    for i in range(1, 3, 1):

        my_grid.rotate_grid(1)
        answer = my_grid.score_grid()
        answer.angle = i

        all_answers.append(answer.model_dump())

        print(i)

    df = pd.DataFrame(all_answers)
    df.to_csv("simulation_results.csv", index=False)


if __name__ == "__main__":
    main()
