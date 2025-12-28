import json
import math
import random
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.data_generator import ProblemInstance


def calculate_tour_distance(tour, distance_matrix):
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += distance_matrix[tour[i]][tour[(i + 1) % n]]
    return total


def two_opt_swap(tour, i, j):
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour


def generate_neighbor(tour):
    n = len(tour)
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, n - 1)
    return two_opt_swap(tour, i, j)


def generate_initial_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour


def nearest_neighbor_tour(distance_matrix):
    n = len(distance_matrix)
    visited = [False] * n
    tour = [0]
    visited[0] = True

    for _ in range(n - 1):
        current = tour[-1]
        nearest = None
        nearest_dist = float("inf")

        for city in range(n):
            if not visited[city] and distance_matrix[current][city] < nearest_dist:
                nearest = city
                nearest_dist = distance_matrix[current][city]

        tour.append(nearest)
        visited[nearest] = True

    return tour


def simulated_annealing(
    instance,
    initial_temp=10000,
    cooling_rate=0.9995,
    min_temp=1e-8,
    max_iterations=None,
    use_nearest_neighbor=True,
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    distance_matrix = instance.distance_matrix
    n = instance.num_cities

    if use_nearest_neighbor:
        current_tour = nearest_neighbor_tour(distance_matrix)
    else:
        current_tour = generate_initial_tour(n)

    current_distance = calculate_tour_distance(current_tour, distance_matrix)

    best_tour = current_tour.copy()
    best_distance = current_distance

    temp = initial_temp
    iteration = 0

    while temp > min_temp:
        if max_iterations and iteration >= max_iterations:
            break

        neighbor_tour = generate_neighbor(current_tour)
        neighbor_distance = calculate_tour_distance(neighbor_tour, distance_matrix)

        delta = neighbor_distance - current_distance

        if delta < 0:
            current_tour = neighbor_tour
            current_distance = neighbor_distance

            if current_distance < best_distance:
                best_tour = current_tour.copy()
                best_distance = current_distance
        else:
            acceptance_prob = math.exp(-delta / temp)
            if random.random() < acceptance_prob:
                current_tour = neighbor_tour
                current_distance = neighbor_distance

        temp *= cooling_rate
        iteration += 1

    return best_tour, best_distance, iteration


def solve_instance(
    instance,
    initial_temp=10000,
    cooling_rate=0.9995,
    min_temp=1e-8,
    max_iterations=None,
    num_runs=5,
    seed=None,
):
    best_tour = None
    best_distance = float("inf")
    all_distances = []
    total_iterations = 0

    for run in range(num_runs):
        run_seed = seed + run if seed else None
        tour, distance, iterations = simulated_annealing(
            instance,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            min_temp=min_temp,
            max_iterations=max_iterations,
            seed=run_seed,
        )
        all_distances.append(distance)
        total_iterations += iterations

        if distance < best_distance:
            best_distance = distance
            best_tour = tour

    return {
        "best_tour": best_tour,
        "best_distance": best_distance,
        "all_distances": all_distances,
        "avg_distance": sum(all_distances) / len(all_distances),
        "total_iterations": total_iterations,
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


def solve_and_save(instance, output_dir, optimal_distance=None, **sa_params):
    print(f"\nSolving {instance.name} ({instance.num_cities} cities) with SA...")

    start_time = time.time()
    result = solve_instance(instance, **sa_params)
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
        "solver": "simulated_annealing",
        "best_tour": result["best_tour"],
        "best_distance": result["best_distance"],
        "avg_distance": result["avg_distance"],
        "all_distances": result["all_distances"],
        "total_iterations": result["total_iterations"],
        "computation_time_seconds": elapsed_time,
        "parameters": sa_params,
    }

    if optimal_distance:
        output["optimal_distance"] = optimal_distance
        output["approximation_ratio"] = ratio

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = get_unique_filepath(output_path, f"{instance.name}_sa", ".json")
    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved to: {result_file}")
    return output


def load_optimal_distances(brute_force_dir):
    summary_file = Path(results_dir) / "optimal_solutions_summary.json"
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
    print("Simulated Annealing TSP Solver")
    print("=" * 60)

    all_results = {}
    total_start = time.time()

    params = {"initial_temp": 10000, "cooling_rate": 0.9995, "num_runs": 5}

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

    summary_file = get_unique_filepath(Path(results_dir), "sa_results_summary", ".json")
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
    results_dir = base_dir / "results" / "sa"
    solve_all_instances(str(data_dir), str(results_dir))
