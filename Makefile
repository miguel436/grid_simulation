.PHONY: install clean lint format

install:
	@echo ">> Installing dependencies"
	poetry install --no-cache

## Lock dependencies
lock:
	poetry lock --no-update

rotation:
	poetry run python src/entrypoints/rotation_plot.py --angle $(angle)

decay:
	poetry run python src/entrypoints/decay_plot.py --radius $(radius)

simulation:
	poetry run python src/entrypoints/run_sim.py

format: clean
	poetry run isort .
	poetry run ruff check . --fix
	poetry run black .