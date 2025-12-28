import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class City:
    id: int
    x: float
    y: float

    def distance_to(self, other: "City") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def to_dict(self) -> dict:
        return {"id": self.id, "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict) -> "City":
        return cls(id=data["id"], x=data["x"], y=data["y"])


@dataclass
class ProblemInstance:
    cities: List[City]
    distance_matrix: List[List[float]] = field(default_factory=list)
    name: str = ""

    def __post_init__(self):
        if not self.distance_matrix and self.cities:
            self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> None:
        n = len(self.cities)
        self.distance_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.cities[i].distance_to(self.cities[j])
                self.distance_matrix[i][j] = dist
                self.distance_matrix[j][i] = dist

    def get_distance(self, city_i: int, city_j: int) -> float:
        return self.distance_matrix[city_i][city_j]

    def get_tour_distance(self, tour: List[int]) -> float:
        total = 0.0
        n = len(tour)
        for i in range(n):
            total += self.distance_matrix[tour[i]][tour[(i + 1) % n]]
        return total

    @property
    def num_cities(self) -> int:
        return len(self.cities)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_cities": self.num_cities,
            "cities": [city.to_dict() for city in self.cities],
            "distance_matrix": self.distance_matrix,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProblemInstance":
        cities = [City.from_dict(c) for c in data["cities"]]
        return cls(
            cities=cities,
            distance_matrix=data.get("distance_matrix", []),
            name=data.get("name", ""),
        )

    def save_to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved problem instance to {filepath}")

    @classmethod
    def load_from_json(cls, filepath: str) -> "ProblemInstance":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def generate_random_cities(
    n: int,
    min_coord: float = 0.0,
    max_coord: float = 1000.0,
    seed: Optional[int] = None,
) -> List[City]:
    if seed is not None:
        random.seed(seed)
    cities = []
    for i in range(n):
        x = random.uniform(min_coord, max_coord)
        y = random.uniform(min_coord, max_coord)
        cities.append(City(id=i, x=x, y=y))
    return cities


def generate_problem_instance(
    n: int,
    name: str = "",
    min_coord: float = 0.0,
    max_coord: float = 1000.0,
    seed: Optional[int] = None,
) -> ProblemInstance:
    cities = generate_random_cities(n, min_coord, max_coord, seed)
    return ProblemInstance(cities=cities, name=name)


def generate_all_datasets(output_dir: str = None, base_seed: int = 42) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = [10, 15, 30, 50, 100, 200]

    print("=" * 60)
    print("Generating TSP Problem Instances")
    print("=" * 60)

    seed_counter = base_seed
    for n in sizes:
        instance_name = f"tsp_{n}"
        filepath = output_dir / f"{instance_name}.json"
        instance = generate_problem_instance(
            n=n, name=instance_name, seed=seed_counter
        )
        seed_counter += 1
        instance.save_to_json(str(filepath))
        print(f"  - {instance_name}: {n} cities")

    print("\n" + "=" * 60)
    print("All datasets generated successfully!")
    print("=" * 60)


def load_all_datasets(data_dir: str = None) -> dict:
    if data_dir is None:
        data_dir = Path(__file__).parent
    else:
        data_dir = Path(data_dir)
    instances = {}
    for json_file in data_dir.glob("tsp_*.json"):
        instance = ProblemInstance.load_from_json(str(json_file))
        instances[instance.name] = instance
        print(f"Loaded: {instance.name} ({instance.num_cities} cities)")
    return instances


if __name__ == "__main__":
    generate_all_datasets()
