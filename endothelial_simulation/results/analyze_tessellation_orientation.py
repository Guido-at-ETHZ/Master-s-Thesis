"""
Recreate tessellation from saved simulation data and test different orientation approaches.
This allows testing the new biological orientation approach without modifying the simulation code.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
import pickle

def load_simulation_data(filepath):
    """
    Load simulation data and extract final cell states.
    """
    print(f"Loading simulation data from: {filepath}")

    try:
        data = np.load(filepath, allow_pickle=True)
        history = data['history'].item()

        print(f"✅ Loaded simulation with {len(history['time'])} time points")
        print(f"Final time: {history['time'][-1]:.1f} minutes")

        # Extract final state
        final_state = {
            'time': history['time'][-1],
            'input_value': history['input_value'][-1],
            'cell_properties': history['cell_properties'][-1]
        }

        # Debug: Check what's in the data file
        print(f"Keys in data file: {list(data.keys())}")

        # Extract grid configuration (handle different data formats)
        grid_config = {'width': 1024, 'height': 1024, 'computation_scale': 4}  # Defaults

        # Try to get config parameters
        if 'config_params' in data:
            try:
                config_params = data['config_params']
                if hasattr(config_params, 'item'):  # If it's a numpy array
                    config_params = config_params.item()

                print(f"Config params type: {type(config_params)}")
                print(f"Config params: {config_params}")

                # Extract grid size if available
                if isinstance(config_params, dict):
                    if 'grid_size' in config_params:
                        grid_size = config_params['grid_size']
                        if isinstance(grid_size, (list, tuple, np.ndarray)) and len(grid_size) >= 2:
                            grid_config['width'] = int(grid_size[0])
                            grid_config['height'] = int(grid_size[1])

            except Exception as e:
                print(f"Warning: Could not parse config_params: {e}")

        print(f"Grid configuration: {grid_config['width']}x{grid_config['height']}")
        print(f"Number of cells: {len(final_state['cell_properties']['areas'])}")

        return final_state, grid_config

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def recreate_cell_positions(final_state, grid_config):
    """
    Recreate cell positions from the simulation data.
    Since we don't have original positions, we'll estimate them.
    """
    cell_props = final_state['cell_properties']
    n_cells = len(cell_props['areas'])

    print(f"Recreating positions for {n_cells} cells...")

    # Generate reasonable cell positions (since originals aren't saved)
    # Use Poisson disk sampling for realistic distribution
    positions = generate_poisson_disk_positions(
        n_cells,
        grid_config['width'],
        grid_config['height']
    )

    # Create cell data structure
    cells_data = []
    for i in range(n_cells):
        cell_data = {
            'id': i,
            'position': positions[i],
            'target_orientation': np.radians(cell_props['target_orientations_degrees'][i]),
            'target_area': cell_props['target_areas'][i],
            'target_aspect_ratio': cell_props['target_aspect_ratios'][i],
            'is_senescent': cell_props['is_senescent'][i],
            'senescence_cause': cell_props['senescence_causes'][i]
        }
        cells_data.append(cell_data)

    return cells_data

def generate_poisson_disk_positions(n_cells, width, height, min_distance_factor=0.6):
    """
    Generate cell positions using Poisson disk sampling.
    """
    area = width * height
    min_dist = np.sqrt(area / n_cells) * min_distance_factor

    positions = []
    attempts = 0
    max_attempts = n_cells * 100

    while len(positions) < n_cells and attempts < max_attempts:
        x = np.random.uniform(min_dist, width - min_dist)
        y = np.random.uniform(min_dist, height - min_dist)
        pos = (x, y)

        # Check distance to existing positions
        valid = True
        for existing_pos in positions:
            dist = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
            if dist < min_dist:
                valid = False
                break

        if valid:
            positions.append(pos)

        attempts += 1

    # Fill remaining positions randomly if needed
    while len(positions) < n_cells:
        x = np.random.uniform(20, width - 20)
        y = np.random.uniform(20, height - 20)
        positions.append((x, y))

    return positions

def create_voronoi_tessellation(cells_data, grid_config):
    """
    Create Voronoi tessellation from cell positions.
    """
    positions = np.array([cell['position'] for cell in cells_data])

    # Create Voronoi diagram
    vor = Voronoi(positions)

    # Get computational grid dimensions
    comp_width = grid_config['width'] // grid_config['computation_scale']
    comp_height = grid_config['height'] // grid_config['computation_scale']

    # Create pixel grid for computational coordinates
    y_coords, x_coords = np.mgrid[0:comp_height, 0:comp_width]
    pixel_coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])

    # Convert positions to computational coordinates
    comp_positions = positions / grid_config['computation_scale']

    # Assign pixels to nearest seed point (simple Voronoi)
    distances = cdist(pixel_coords, comp_positions)
    nearest_seed_indices = np.argmin(distances, axis=1)

    # Create territories for each cell
    territories = {}
    for i, cell in enumerate(cells_data):
        pixel_indices = np.where(nearest_seed_indices == i)[0]
        territory_pixels = [(int(pixel_coords[idx][0]), int(pixel_coords[idx][1]))
                           for idx in pixel_indices]
        territories[cell['id']] = territory_pixels
        cell['territory_pixels'] = territory_pixels

    print(f"Created tessellation with {len(territories)} territories")
    return territories

def calculate_pca_orientation(territory_pixels):
    """
    Calculate orientation using PCA (current method).
    """
    if len(territory_pixels) < 3:
        return 0.0

    pixels_array = np.array(territory_pixels)
    centroid = np.mean(pixels_array, axis=0)
    centered = pixels_array - centroid

    try:
        cov_matrix = np.cov(centered, rowvar=False)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue
        idx = np.argsort(eigenvals)[::-1]
        principal_axis = eigenvecs[:, idx[0]]
        orientation = np.arctan2(principal_axis[1], principal_axis[0])
        return orientation
    except:
        return 0.0

def calculate_biological_orientation(cell_data):
    """
    Calculate orientation using biological approach (new method).
    """
    return cell_data['target_orientation']

def compare_orientation_methods(cells_data):
    """
    Compare PCA vs biological orientation methods.
    """
    print("\nCOMPARING ORIENTATION METHODS")
    print("=" * 50)

    pca_orientations = []
    bio_orientations = []

    for cell in cells_data:
        # Calculate PCA orientation from territory
        pca_orient = calculate_pca_orientation(cell['territory_pixels'])
        pca_orientations.append(pca_orient)

        # Use biological orientation
        bio_orient = calculate_biological_orientation(cell)
        bio_orientations.append(bio_orient)

        cell['pca_orientation'] = pca_orient
        cell['bio_orientation'] = bio_orient

    # Convert to degrees for analysis
    pca_degrees = [np.degrees(angle) for angle in pca_orientations]
    bio_degrees = [np.degrees(angle) for angle in bio_orientations]

    print(f"PCA Method (current):")
    print(f"  Mean: {np.mean(pca_degrees):.1f}°")
    print(f"  Std:  {np.std(pca_degrees):.1f}°")
    print(f"  Range: {np.min(pca_degrees):.1f}° to {np.max(pca_degrees):.1f}°")

    print(f"\nBiological Method (proposed):")
    print(f"  Mean: {np.mean(bio_degrees):.1f}°")
    print(f"  Std:  {np.std(bio_degrees):.1f}°")
    print(f"  Range: {np.min(bio_degrees):.1f}° to {np.max(bio_degrees):.1f}°")

    # Calculate difference
    differences = [abs(p - b) for p, b in zip(pca_degrees, bio_degrees)]
    print(f"\nDifference (PCA - Biological):")
    print(f"  Mean difference: {np.mean(differences):.1f}°")
    print(f"  Max difference:  {np.max(differences):.1f}°")

    return pca_orientations, bio_orientations

def create_comparison_visualization(cells_data, grid_config, final_state):
    """
    Create side-by-side visualization comparing both orientation methods.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Color mapping
    color_map = {'healthy': 'green', 'telomere': 'red', 'stress': 'blue'}

    for plot_idx, (ax, method_name, orient_key) in enumerate([
        (axes[0], 'PCA Method (Current)', 'pca_orientation'),
        (axes[1], 'Biological Method (Proposed)', 'bio_orientation'),
        (axes[2], 'Flow Alignment Comparison', 'both')
    ]):

        ax.set_xlim(0, grid_config['width'])
        ax.set_ylim(0, grid_config['height'])
        ax.set_aspect('equal')
        ax.set_title(method_name, fontsize=14)

        if plot_idx < 2:  # Individual method plots
            # Plot territories and orientations
            for cell in cells_data:
                # Get territory in display coordinates
                display_pixels = [(x * grid_config['computation_scale'],
                                  y * grid_config['computation_scale'])
                                 for x, y in cell['territory_pixels']]

                if len(display_pixels) < 3:
                    continue

                # Create polygon
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(display_pixels)
                    hull_points = np.array(display_pixels)[hull.vertices]

                    # Determine color
                    if not cell['is_senescent']:
                        color = color_map['healthy']
                        alpha = 0.6
                    elif cell['senescence_cause'] == 'telomere':
                        color = color_map['telomere']
                        alpha = 0.8
                    else:
                        color = color_map['stress']
                        alpha = 0.8

                    polygon = Polygon(hull_points, facecolor=color, alpha=alpha,
                                     edgecolor='black', linewidth=0.5)
                    ax.add_patch(polygon)

                    # Calculate centroid in display coordinates
                    centroid = np.mean(display_pixels, axis=0)

                    # Draw orientation vector
                    orientation = cell[orient_key]
                    vector_length = 30  # Fixed length for visibility
                    dx = vector_length * np.cos(orientation)
                    dy = vector_length * np.sin(orientation)

                    ax.arrow(centroid[0], centroid[1], dx, dy,
                            head_width=vector_length*0.2, head_length=vector_length*0.2,
                            fc='white', ec='black', alpha=0.9, width=vector_length*0.05)

                except Exception as e:
                    # Fallback: scatter plot
                    display_pixels_array = np.array(display_pixels)
                    ax.scatter(display_pixels_array[:, 0], display_pixels_array[:, 1],
                              c=color, alpha=alpha, s=1, marker='s')

        else:  # Comparison plot (flow alignment)
            # Convert orientations to flow alignment (0-90°)
            pca_flow = []
            bio_flow = []

            for cell in cells_data:
                # Convert PCA orientation to flow alignment
                pca_deg = np.degrees(cell['pca_orientation'])
                pca_alignment = min(abs(pca_deg) % 180, 180 - abs(pca_deg) % 180)
                pca_flow.append(pca_alignment)

                # Convert biological orientation to flow alignment
                bio_deg = np.degrees(cell['bio_orientation'])
                bio_alignment = min(abs(bio_deg) % 180, 180 - abs(bio_deg) % 180)
                bio_flow.append(bio_alignment)

            # Create histogram comparison
            ax.hist(pca_flow, bins=15, alpha=0.6, color='red',
                   label=f'PCA (mean: {np.mean(pca_flow):.1f}°)', range=(0, 90))
            ax.hist(bio_flow, bins=15, alpha=0.6, color='blue',
                   label=f'Biological (mean: {np.mean(bio_flow):.1f}°)', range=(0, 90))

            ax.set_xlabel('Flow Alignment Angle (degrees)')
            ax.set_ylabel('Number of Cells')
            ax.set_xlim(0, 90)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add reference lines
            ax.axvline(0, color='green', linestyle='-', alpha=0.8, linewidth=2, label='Perfect alignment')
            ax.axvline(45, color='orange', linestyle='--', alpha=0.6)
            ax.axvline(90, color='red', linestyle='-', alpha=0.8, linewidth=2, label='Perpendicular')

    # Add legend to first two plots
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', alpha=0.6, label='Healthy'),
        Patch(facecolor='red', edgecolor='black', alpha=0.8, label='Telomere-Senescent'),
        Patch(facecolor='blue', edgecolor='black', alpha=0.8, label='Stress-Senescent')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Add flow direction indicator
    for ax in axes[:2]:
        arrow_length = grid_config['width'] * 0.08
        arrow_x = grid_config['width'] * 0.5
        arrow_y = grid_config['height'] * 0.05

        ax.arrow(arrow_x - arrow_length/2, arrow_y, arrow_length, 0,
                head_width=arrow_length*0.3, head_length=arrow_length*0.2,
                fc='black', ec='black', width=arrow_length*0.08)
        ax.text(arrow_x, arrow_y - arrow_length*0.5, "Flow Direction",
               ha='center', va='top', fontsize=10, weight='bold')

    # Add simulation info
    info_text = (f"Simulation: {final_state['time']:.0f} min\n"
                f"Shear Stress: {final_state['input_value']:.2f} Pa\n"
                f"Cells: {len(cells_data)}")

    axes[0].text(0.02, 0.98, info_text, transform=axes[0].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    return fig

def analyze_results(cells_data, final_state):
    """
    Analyze and summarize the results.
    """
    print("\nRESULTS ANALYSIS")
    print("=" * 30)

    # Convert orientations to flow alignment for analysis
    pca_flow_alignments = []
    bio_flow_alignments = []

    for cell in cells_data:
        # PCA flow alignment
        pca_deg = np.degrees(cell['pca_orientation'])
        pca_alignment = min(abs(pca_deg) % 180, 180 - abs(pca_deg) % 180)
        pca_flow_alignments.append(pca_alignment)

        # Biological flow alignment
        bio_deg = np.degrees(cell['bio_orientation'])
        bio_alignment = min(abs(bio_deg) % 180, 180 - abs(bio_deg) % 180)
        bio_flow_alignments.append(bio_alignment)

    shear_stress = final_state['input_value']

    print(f"Shear Stress: {shear_stress:.2f} Pa")
    print(f"Expected: Cells should be well-aligned (close to 0°) at this stress level")

    print(f"\nPCA Method Results:")
    print(f"  Mean flow alignment: {np.mean(pca_flow_alignments):.1f}°")
    print(f"  Cells well-aligned (< 30°): {sum(1 for a in pca_flow_alignments if a < 30)}/{len(pca_flow_alignments)}")
    print(f"  Assessment: {'✅ Good' if np.mean(pca_flow_alignments) < 30 else '❌ Poor'} alignment")

    print(f"\nBiological Method Results:")
    print(f"  Mean flow alignment: {np.mean(bio_flow_alignments):.1f}°")
    print(f"  Cells well-aligned (< 30°): {sum(1 for a in bio_flow_alignments if a < 30)}/{len(bio_flow_alignments)}")
    print(f"  Assessment: {'✅ Good' if np.mean(bio_flow_alignments) < 30 else '❌ Poor'} alignment")

    print(f"\nRecommendation:")
    if np.mean(bio_flow_alignments) < np.mean(pca_flow_alignments):
        print("✅ Biological method gives better flow alignment")
        print("✅ Recommended: Switch to biological orientation approach")
    else:
        print("❓ PCA method performs similarly or better")
        print("❓ Consider other factors or check biological targets")

def main(filepath="simulation_20250611-143910.npz"):
    """
    Main function to recreate tessellation and test orientation methods.
    """
    print("TESSELLATION RECREATION AND ORIENTATION METHOD TESTING")
    print("=" * 70)

    # Load simulation data
    final_state, grid_config = load_simulation_data(filepath)
    if final_state is None:
        print("❌ Failed to load simulation data. Cannot proceed.")
        return None, None

    # Recreate cell positions and properties
    cells_data = recreate_cell_positions(final_state, grid_config)

    # Create tessellation
    territories = create_voronoi_tessellation(cells_data, grid_config)

    # Compare orientation methods
    pca_orientations, bio_orientations = compare_orientation_methods(cells_data)

    # Create visualization
    fig = create_comparison_visualization(cells_data, grid_config, final_state)

    # Save results
    plt.savefig('orientation_method_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison saved as 'orientation_method_comparison.png'")

    # Analyze results
    analyze_results(cells_data, final_state)

    plt.show()

    return cells_data, fig

if __name__ == "__main__":
    # Run with your simulation file
    result = main("simulation_20250611-143910.npz")
    if result is not None:
        cells_data, fig = result
        print("✅ Analysis completed successfully!")