import json
import time
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.data_generator import ProblemInstance


def brute_force_tsp(instance):
    n = instance.num_cities

    if n <= 1:
        return [0] if n == 1 else [], 0.0

    if n == 2:
        return [0, 1], instance.get_distance(0, 1) * 2

    remaining_cities = list(range(1, n))
    best_tour = None
    best_distance = float("inf")

    for perm in permutations(remaining_cities):
        tour = [0] + list(perm)
        distance = instance.get_tour_distance(tour)

        if distance < best_distance:
            best_distance = distance
            best_tour = tour

    return best_tour, best_distance


def brute_force_tsp_recursive(instance):
    n = instance.num_cities

    if n <= 1:
        return [0] if n == 1 else [], 0.0

    if n == 2:
        return [0, 1], instance.get_distance(0, 1) * 2

    best_tour = None
    best_distance = float("inf")

    def recurse(current_tour, visited, current_distance):
        nonlocal best_tour, best_distance

        if current_distance >= best_distance:
            return

        if len(current_tour) == n:
            total_distance = current_distance + instance.get_distance(
                current_tour[-1], current_tour[0]
            )
            if total_distance < best_distance:
                best_distance = total_distance
                best_tour = current_tour.copy()
            return

        last_city = current_tour[-1]
        for next_city in range(n):
            if next_city not in visited:
                visited.add(next_city)
                current_tour.append(next_city)
                new_distance = current_distance + instance.get_distance(
                    last_city, next_city
                )
                recurse(current_tour, visited, new_distance)
                current_tour.pop()
                visited.remove(next_city)

    recurse([0], {0}, 0.0)
    return best_tour, best_distance


def solve_and_save_optimal(instance, output_dir, use_recursive=True, max_cities=15):
    n = instance.num_cities

    if n > max_cities:
        print(f"Skipping {instance.name} - too many cities ({n} > {max_cities})")
        return None

    print(f"\nSolving {instance.name} ({n} cities)...")

    solver = brute_force_tsp_recursive if use_recursive else brute_force_tsp
    solver_name = "recursive" if use_recursive else "itertools"

    start_time = time.time()
    best_tour, best_distance = solver(instance)
    elapsed_time = time.time() - start_time

    print(f"  Optimal distance: {best_distance:.2f}")
    print(f"  Optimal tour: {best_tour}")
    print(f"  Time ({solver_name}): {elapsed_time:.4f} seconds")

    result = {
        "instance_name": instance.name,
        "num_cities": n,
        "optimal_tour": best_tour,
        "optimal_distance": best_distance,
        "solver": f"brute_force_{solver_name}",
        "computation_time_seconds": elapsed_time,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / f"{instance.name}_optimal.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved to: {result_file}")
    return result


def solve(data_dir, results_dir):
    data_path = Path(data_dir)
    results = {}

    print("=" * 60)
    print("Brute Force TSP Solver - Finding Optimal Solutions")
    print("=" * 60)

    small_files = sorted([f for f in data_path.glob("tsp_*.json") if int(f.stem.split("_")[1]) <= 15])

    if not small_files:
        print(f"No small instance files (N <= 15) found in {data_dir}")
        return results

    total_start = time.time()

    for json_file in small_files:
        instance = ProblemInstance.load_from_json(str(json_file))
        result = solve_and_save_optimal(instance, results_dir)
        if result:
            results[instance.name] = result

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Solved {len(results)} instances in {total_time:.2f} seconds")

    for name, res in results.items():
        print(f"  {name}: {res['optimal_distance']:.2f} ({res['computation_time_seconds']:.4f}s)")

    summary_file = Path(results_dir) / "optimal_solutions_summary.json"
    summary = {
        "total_instances": len(results),
        "total_time_seconds": total_time,
        "results": results,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    return results


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results" / "brute_force"
    solve(str(data_dir), str(results_dir))
