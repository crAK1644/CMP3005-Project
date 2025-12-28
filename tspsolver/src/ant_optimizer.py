import json
import random
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.data_generator import ProblemInstance


class AntColonyOptimizer:
    def __init__(
        self,
        instance,
        num_ants=None,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q=100,
        initial_pheromone=1.0,
        max_iterations=100,
        use_elitist=True,
        seed=None,
    ):
        if seed is not None:
            random.seed(seed)

        self.instance = instance
        self.n = instance.num_cities
        self.distance_matrix = instance.distance_matrix

        self.num_ants = num_ants if num_ants else self.n
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.use_elitist = use_elitist
        self.max_iterations = max_iterations

        self.pheromone = [
            [initial_pheromone for _ in range(self.n)] for _ in range(self.n)
        ]

        self.heuristic = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.distance_matrix[i][j] > 0:
                    self.heuristic[i][j] = 1.0 / self.distance_matrix[i][j]

        self.best_tour = None
        self.best_distance = float("inf")

    def calculate_tour_distance(self, tour):
        total = 0.0
        for i in range(len(tour)):
            total += self.distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return total

    def select_next_city(self, current_city, visited):
        probabilities = []
        total = 0.0

        for city in range(self.n):
            if city not in visited:
                tau = self.pheromone[current_city][city] ** self.alpha
                eta = self.heuristic[current_city][city] ** self.beta
                prob = tau * eta
                probabilities.append((city, prob))
                total += prob

        if total == 0:
            unvisited = [c for c in range(self.n) if c not in visited]
            return random.choice(unvisited) if unvisited else None

        r = random.random() * total
        cumulative = 0.0
        for city, prob in probabilities:
            cumulative += prob
            if cumulative >= r:
                return city

        return probabilities[-1][0] if probabilities else None

    def construct_solution(self, start_city=None):
        if start_city is None:
            start_city = random.randint(0, self.n - 1)

        tour = [start_city]
        visited = {start_city}

        while len(tour) < self.n:
            current = tour[-1]
            next_city = self.select_next_city(current, visited)
            if next_city is None:
                break
            tour.append(next_city)
            visited.add(next_city)

        return tour

    def evaporate_pheromone(self):
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] *= (1 - self.rho)

    def deposit_pheromone(self, tour, distance):
        deposit = self.q / distance
        for i in range(len(tour)):
            city_a = tour[i]
            city_b = tour[(i + 1) % len(tour)]
            self.pheromone[city_a][city_b] += deposit
            self.pheromone[city_b][city_a] += deposit

    def run_iteration(self):
        iteration_best_tour = None
        iteration_best_distance = float("inf")

        for ant in range(self.num_ants):
            tour = self.construct_solution()
            distance = self.calculate_tour_distance(tour)

            if distance < iteration_best_distance:
                iteration_best_distance = distance
                iteration_best_tour = tour

            if distance < self.best_distance:
                self.best_distance = distance
                self.best_tour = tour.copy()

        self.evaporate_pheromone()

        if self.use_elitist and self.best_tour:
            self.deposit_pheromone(self.best_tour, self.best_distance)
        else:
            self.deposit_pheromone(iteration_best_tour, iteration_best_distance)

        return iteration_best_tour, iteration_best_distance

    def solve(self):
        for iteration in range(self.max_iterations):
            self.run_iteration()

        return self.best_tour, self.best_distance


def ant_colony_optimization(
    instance,
    num_ants=None,
    alpha=1.0,
    beta=2.0,
    rho=0.1,
    q=100,
    initial_pheromone=1.0,
    max_iterations=100,
    use_elitist=True,
    seed=None,
):
    aco = AntColonyOptimizer(
        instance=instance,
        num_ants=num_ants,
        alpha=alpha,
        beta=beta,
        rho=rho,
        q=q,
        initial_pheromone=initial_pheromone,
        max_iterations=max_iterations,
        use_elitist=use_elitist,
        seed=seed,
    )
    return aco.solve()


def solve_instance(
    instance,
    num_ants=None,
    alpha=1.0,
    beta=2.0,
    rho=0.1,
    q=100,
    initial_pheromone=1.0,
    max_iterations=100,
    use_elitist=True,
    num_runs=5,
    seed=None,
):
    best_tour = None
    best_distance = float("inf")
    all_distances = []

    for run in range(num_runs):
        run_seed = seed + run if seed else None
        tour, distance = ant_colony_optimization(
            instance,
            num_ants=num_ants,
            alpha=alpha,
            beta=beta,
            rho=rho,
            q=q,
            initial_pheromone=initial_pheromone,
            max_iterations=max_iterations,
            use_elitist=use_elitist,
            seed=run_seed,
        )
        all_distances.append(distance)

        if distance < best_distance:
            best_distance = distance
            best_tour = tour

    return {
        "best_tour": best_tour,
        "best_distance": best_distance,
        "all_distances": all_distances,
        "avg_distance": sum(all_distances) / len(all_distances),
    }


def get_unique_filepath(output_path, base_name, extension):
    filepath = output_path / f"{base_name}{extension}"
    if not filepath.exists():
        return filepath
    
    counter = 1
    while True:
        filepath = output_path / f"{base_name}_{counter}{extension}"
        if not filepath.exists():
            return filepath
        counter += 1


def solve_and_save(instance, output_dir, optimal_distance=None, **aco_params):
    print(f"\nSolving {instance.name} ({instance.num_cities} cities) with ACO...")

    start_time = time.time()
    result = solve_instance(instance, **aco_params)
    elapsed_time = time.time() - start_time

    print(f"  Best distance: {result['best_distance']:.2f}")
    print(f"  Avg distance: {result['avg_distance']:.2f}")
    print(f"  Time: {elapsed_time:.4f} seconds")

    if optimal_distance:
        ratio = result["best_distance"] / optimal_distance
        print(f"  Approximation ratio: {ratio:.4f}")
    else:
        ratio = None

    output = {
        "instance_name": instance.name,
        "num_cities": instance.num_cities,
        "solver": "ant_colony_optimization",
        "best_tour": result["best_tour"],
        "best_distance": result["best_distance"],
        "avg_distance": result["avg_distance"],
        "all_distances": result["all_distances"],
        "computation_time_seconds": elapsed_time,
        "parameters": {
            k: v for k, v in aco_params.items() if k != "seed"
        },
    }

    if optimal_distance:
        output["optimal_distance"] = optimal_distance
        output["approximation_ratio"] = ratio

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = get_unique_filepath(output_path, f"{instance.name}_aco", ".json")
    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved to: {result_file}")
    return output


def load_optimal_distances(brute_force_dir):
    summary_file = Path(brute_force_dir) / "optimal_solutions_summary.json"
    if not summary_file.exists():
        return {}

    with open(summary_file, "r") as f:
        summary = json.load(f)

    optimal = {}
    for name, data in summary.get("results", {}).items():
        optimal[name] = data["optimal_distance"]

    return optimal


def solve_all_instances(data_dir, results_dir):
    data_path = Path(data_dir)
    brute_force_dir = Path(results_dir).parent / "brute_force"
    optimal_distances = load_optimal_distances(brute_force_dir)

    print("=" * 60)
    print("Ant Colony Optimization TSP Solver")
    print("=" * 60)

    all_results = {}
    total_start = time.time()

    params = {
        "alpha": 1.0, "beta": 3.0, "rho": 0.1, "q": 100,
        "max_iterations": 150, "num_runs": 3
    }

    files = sorted(data_path.glob("tsp_*.json"))
    for json_file in files:
        instance = ProblemInstance.load_from_json(str(json_file))
        optimal = optimal_distances.get(instance.name)

        result = solve_and_save(
            instance, results_dir, optimal_distance=optimal, **params
        )
        all_results[instance.name] = result

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total time: {total_time:.2f} seconds")

    for name, res in all_results.items():
        line = f"  {name}: {res['best_distance']:.2f}"
        if "approximation_ratio" in res:
            line += f" (ratio: {res['approximation_ratio']:.4f})"
        print(line)

    summary_file = get_unique_filepath(Path(results_dir), "aco_results_summary", ".json")
    with open(summary_file, "w") as f:
        json.dump(
            {
                "total_time_seconds": total_time,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to: {summary_file}")

    return all_results


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results" / "aco"
    solve_all_instances(str(data_dir), str(results_dir))
