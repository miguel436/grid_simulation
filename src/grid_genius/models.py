from typing import List

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


class RayGrid(BaseModel):
    min_grid: float
    max_grid: float
    ray_grid: List[float]


class GridScore(BaseModel):
    mean: float
    median: float
    stdev: float
    zeros: int
    scores: List[float]
    angle: float = Field(default=0)
