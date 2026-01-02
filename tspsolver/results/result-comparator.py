"""
TSP Solver Results Comparator
Compares results from ACO (Ant Colony Optimization) and SA (Simulated Annealing)
algorithms using various visualizations with matplotlib.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set the style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define paths
RESULTS_DIR = Path(__file__).parent
ACO_DIR = RESULTS_DIR / "aco"
SA_DIR = RESULTS_DIR / "sa"
BF_DIR = RESULTS_DIR / "brute_force"


def load_json(filepath):
    """Load JSON file and return data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_all_results():
    """Load all result summary files."""
    aco_data = load_json(ACO_DIR / "aco_results_summary.json")
    sa_data = load_json(SA_DIR / "sa_results_summary.json")
    bf_data = load_json(BF_DIR / "optimal_solutions_summary.json")
    return aco_data, sa_data, bf_data


def extract_comparison_data(aco_data, sa_data, bf_data):
    """Extract comparison data from results."""
    # Get all problem sizes
    aco_results = aco_data['results']
    sa_results = sa_data['results']
    bf_results = bf_data['results']
    
    # Get common instances and sort by number of cities
    all_instances = sorted(
        set(aco_results.keys()) & set(sa_results.keys()),
        key=lambda x: int(x.split('_')[1])
    )
    
    comparison = {
        'instances': [],
        'num_cities': [],
        'aco_best': [],
        'aco_avg': [],
        'aco_time': [],
        'sa_best': [],
        'sa_avg': [],
        'sa_time': [],
        'optimal': [],
        'aco_all_distances': [],
        'sa_all_distances': []
    }
    
    for instance in all_instances:
        aco = aco_results[instance]
        sa = sa_results[instance]
        
        comparison['instances'].append(instance)
        comparison['num_cities'].append(aco['num_cities'])
        comparison['aco_best'].append(aco['best_distance'])
        comparison['aco_avg'].append(aco['avg_distance'])
        comparison['aco_time'].append(aco['computation_time_seconds'])
        comparison['sa_best'].append(sa['best_distance'])
        comparison['sa_avg'].append(sa['avg_distance'])
        comparison['sa_time'].append(sa['computation_time_seconds'])
        comparison['aco_all_distances'].append(aco['all_distances'])
        comparison['sa_all_distances'].append(sa['all_distances'])
        
        # Get optimal if available
        if instance in bf_results:
            comparison['optimal'].append(bf_results[instance]['optimal_distance'])
        else:
            comparison['optimal'].append(None)
    
    return comparison


def plot_distance_comparison(comparison):
    """Plot best distances comparison between ACO and SA."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(comparison['instances']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comparison['aco_best'], width, label='ACO Best', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, comparison['sa_best'], width, label='SA Best', color='#3498db', alpha=0.8)
    
    # Add optimal markers where available
    for i, opt in enumerate(comparison['optimal']):
        if opt is not None:
            ax.scatter(i, opt, marker='*', s=200, c='red', zorder=5, label='Optimal' if i == 0 else '')
    
    ax.set_xlabel('Problem Instance', fontsize=12)
    ax.set_ylabel('Best Distance', fontsize=12)
    ax.set_title('TSP Best Distance Comparison: ACO vs SA', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} cities" for c in comparison['num_cities']], rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    return fig


def plot_computation_time(comparison):
    """Plot computation time comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(comparison['instances']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comparison['aco_time'], width, label='ACO', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, comparison['sa_time'], width, label='SA', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Problem Instance', fontsize=12)
    ax.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax.set_title('Computation Time Comparison: ACO vs SA', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} cities" for c in comparison['num_cities']], rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')  # Use log scale for better visibility
    
    plt.tight_layout()
    return fig


def plot_quality_vs_time(comparison):
    """Plot solution quality vs computation time scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize distances by number of cities for fair comparison
    aco_quality = [d / n for d, n in zip(comparison['aco_best'], comparison['num_cities'])]
    sa_quality = [d / n for d, n in zip(comparison['sa_best'], comparison['num_cities'])]
    
    scatter1 = ax.scatter(comparison['aco_time'], aco_quality, 
                          s=[n*5 for n in comparison['num_cities']], 
                          c='#2ecc71', alpha=0.7, label='ACO', edgecolors='black')
    scatter2 = ax.scatter(comparison['sa_time'], sa_quality, 
                          s=[n*5 for n in comparison['num_cities']], 
                          c='#3498db', alpha=0.7, label='SA', edgecolors='black')
    
    # Add labels for each point
    for i, instance in enumerate(comparison['instances']):
        ax.annotate(f"{comparison['num_cities'][i]}c", 
                    (comparison['aco_time'][i], aco_quality[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.annotate(f"{comparison['num_cities'][i]}c", 
                    (comparison['sa_time'][i], sa_quality[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Computation Time (seconds)', fontsize=12)
    ax.set_ylabel('Normalized Distance (distance/cities)', fontsize=12)
    ax.set_title('Solution Quality vs Computation Time\n(bubble size = number of cities)', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_variance_comparison(comparison):
    """Plot variance/consistency comparison using box plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (instance, num_cities) in enumerate(zip(comparison['instances'], comparison['num_cities'])):
        if i >= 6:  # Only show first 6
            break
            
        ax = axes[i]
        aco_distances = comparison['aco_all_distances'][i]
        sa_distances = comparison['sa_all_distances'][i]
        
        bp = ax.boxplot([aco_distances, sa_distances], 
                        tick_labels=['ACO', 'SA'],
                        patch_artist=True)
        
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#3498db')
        
        ax.set_ylabel('Distance')
        ax.set_title(f'{num_cities} Cities')
    
    fig.suptitle('Solution Consistency Comparison (Multiple Runs)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_scalability(comparison):
    """Plot how algorithms scale with problem size."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distance vs Problem Size
    ax1.plot(comparison['num_cities'], comparison['aco_best'], 'o-', 
             color='#2ecc71', label='ACO Best', linewidth=2, markersize=8)
    ax1.plot(comparison['num_cities'], comparison['sa_best'], 's-', 
             color='#3498db', label='SA Best', linewidth=2, markersize=8)
    ax1.plot(comparison['num_cities'], comparison['aco_avg'], 'o--', 
             color='#2ecc71', alpha=0.5, label='ACO Avg', linewidth=1)
    ax1.plot(comparison['num_cities'], comparison['sa_avg'], 's--', 
             color='#3498db', alpha=0.5, label='SA Avg', linewidth=1)
    
    ax1.set_xlabel('Number of Cities', fontsize=12)
    ax1.set_ylabel('Distance', fontsize=12)
    ax1.set_title('Distance vs Problem Size', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Computation Time vs Problem Size
    ax2.plot(comparison['num_cities'], comparison['aco_time'], 'o-', 
             color='#2ecc71', label='ACO', linewidth=2, markersize=8)
    ax2.plot(comparison['num_cities'], comparison['sa_time'], 's-', 
             color='#3498db', label='SA', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of Cities', fontsize=12)
    ax2.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax2.set_title('Computation Time vs Problem Size', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    return fig


def plot_performance_ratio(comparison):
    """Plot performance ratio (ACO/SA) for distance and time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distance ratio
    distance_ratio = [aco/sa for aco, sa in zip(comparison['aco_best'], comparison['sa_best'])]
    colors = ['#2ecc71' if r < 1 else '#e74c3c' for r in distance_ratio]
    
    bars1 = ax1.bar(range(len(comparison['instances'])), distance_ratio, color=colors, alpha=0.8)
    ax1.axhline(y=1, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel('Problem Instance', fontsize=12)
    ax1.set_ylabel('ACO/SA Distance Ratio', fontsize=12)
    ax1.set_title('Distance Ratio (< 1 means ACO is better)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(comparison['instances'])))
    ax1.set_xticklabels([f"{c}c" for c in comparison['num_cities']])
    
    # Time ratio  
    time_ratio = [aco/sa for aco, sa in zip(comparison['aco_time'], comparison['sa_time'])]
    colors = ['#2ecc71' if r < 1 else '#e74c3c' for r in time_ratio]
    
    bars2 = ax2.bar(range(len(comparison['instances'])), time_ratio, color=colors, alpha=0.8)
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Problem Instance', fontsize=12)
    ax2.set_ylabel('ACO/SA Time Ratio', fontsize=12)
    ax2.set_title('Time Ratio (< 1 means ACO is faster)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(comparison['instances'])))
    ax2.set_xticklabels([f"{c}c" for c in comparison['num_cities']])
    ax2.set_yscale('log')
    
    plt.tight_layout()
    return fig


def plot_summary_heatmap(comparison):
    """Create a summary heatmap of normalized performance."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create normalized metrics (0-1 scale, lower is better)
    metrics = ['Best Distance', 'Avg Distance', 'Computation Time', 'Variance']
    algorithms = ['ACO', 'SA']
    
    data = []
    for i in range(len(comparison['instances'])):
        row = []
        # Best distance (normalized)
        min_dist = min(comparison['aco_best'][i], comparison['sa_best'][i])
        row.append(comparison['aco_best'][i] / min_dist)
        row.append(comparison['sa_best'][i] / min_dist)
        
        # Average distance (normalized)
        min_avg = min(comparison['aco_avg'][i], comparison['sa_avg'][i])
        row.append(comparison['aco_avg'][i] / min_avg)
        row.append(comparison['sa_avg'][i] / min_avg)
        
        # Time (normalized)
        min_time = min(comparison['aco_time'][i], comparison['sa_time'][i])
        row.append(comparison['aco_time'][i] / min_time)
        row.append(comparison['sa_time'][i] / min_time)
        
        # Variance (normalized)
        aco_var = np.var(comparison['aco_all_distances'][i])
        sa_var = np.var(comparison['sa_all_distances'][i])
        min_var = min(aco_var, sa_var) if min(aco_var, sa_var) > 0 else 1
        row.append(aco_var / min_var if min_var > 0 else 1)
        row.append(sa_var / min_var if min_var > 0 else 1)
        
        data.append(row)
    
    # Reshape data for heatmap
    data = np.array(data)
    
    # Create labels for columns
    col_labels = [f'{m}\n{a}' for m in metrics for a in algorithms]
    row_labels = [f"{c} cities" for c in comparison['num_cities']]
    
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=2)
    
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Relative Performance (1 = best)', rotation=-90, va="bottom")
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Performance Comparison Heatmap\n(Green = Better, Red = Worse)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def print_summary_statistics(comparison):
    """Print summary statistics to console."""
    print("\n" + "="*60)
    print("TSP SOLVER COMPARISON SUMMARY")
    print("="*60)
    
    print("\n📊 BEST DISTANCES:")
    print("-" * 50)
    print(f"{'Instance':<15} {'ACO':<12} {'SA':<12} {'Winner':<10}")
    print("-" * 50)
    
    aco_wins_dist = 0
    sa_wins_dist = 0
    
    for i, instance in enumerate(comparison['instances']):
        aco = comparison['aco_best'][i]
        sa = comparison['sa_best'][i]
        if aco < sa:
            winner = "ACO ✓"
            aco_wins_dist += 1
        elif sa < aco:
            winner = "SA ✓"
            sa_wins_dist += 1
        else:
            winner = "Tie"
        print(f"{comparison['num_cities'][i]} cities{'':<8} {aco:<12.2f} {sa:<12.2f} {winner:<10}")
    
    print(f"\nDistance wins: ACO={aco_wins_dist}, SA={sa_wins_dist}")
    
    print("\n⏱️ COMPUTATION TIME:")
    print("-" * 50)
    print(f"{'Instance':<15} {'ACO (s)':<12} {'SA (s)':<12} {'Faster':<10}")
    print("-" * 50)
    
    aco_wins_time = 0
    sa_wins_time = 0
    
    for i, instance in enumerate(comparison['instances']):
        aco = comparison['aco_time'][i]
        sa = comparison['sa_time'][i]
        if aco < sa:
            faster = "ACO ✓"
            aco_wins_time += 1
        else:
            faster = "SA ✓"
            sa_wins_time += 1
        print(f"{comparison['num_cities'][i]} cities{'':<8} {aco:<12.4f} {sa:<12.4f} {faster:<10}")
    
    print(f"\nSpeed wins: ACO={aco_wins_time}, SA={sa_wins_time}")
    
    # Overall statistics
    print("\n📈 OVERALL STATISTICS:")
    print("-" * 50)
    print(f"Total ACO time: {sum(comparison['aco_time']):.2f} seconds")
    print(f"Total SA time: {sum(comparison['sa_time']):.2f} seconds")
    print(f"ACO speedup factor: {sum(comparison['sa_time'])/sum(comparison['aco_time']):.2f}x" 
          if sum(comparison['aco_time']) < sum(comparison['sa_time']) 
          else f"SA speedup factor: {sum(comparison['aco_time'])/sum(comparison['sa_time']):.2f}x")
    
    print("="*60)


def main():
    """Main function to generate all comparison plots."""
    print("Loading results...")
    aco_data, sa_data, bf_data = load_all_results()
    
    print("Extracting comparison data...")
    comparison = extract_comparison_data(aco_data, sa_data, bf_data)
    
    # Print summary statistics
    print_summary_statistics(comparison)
    
    print("\nGenerating plots...")
    
    # Generate all plots
    figures = [
        ('1_distance_comparison', plot_distance_comparison(comparison)),
        ('2_computation_time', plot_computation_time(comparison)),
        ('3_quality_vs_time', plot_quality_vs_time(comparison)),
        ('4_variance_comparison', plot_variance_comparison(comparison)),
        ('5_scalability', plot_scalability(comparison)),
        ('6_performance_ratio', plot_performance_ratio(comparison)),
        ('7_summary_heatmap', plot_summary_heatmap(comparison))
    ]
    
    # Save all figures
    output_dir = RESULTS_DIR / "comparison_plots"
    output_dir.mkdir(exist_ok=True)
    
    for name, fig in figures:
        filepath = output_dir / f"{name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filepath}")
    
    print(f"\n✅ All plots saved to: {output_dir}")
    
    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
