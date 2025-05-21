"""
Animation module for endothelial cell mechanotransduction simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import os
import time


def create_metrics_animation(plotter, simulator, metrics=None, save_path=None, fps=10, dpi=100):
    """
    Create an animation showing the evolution of simulation metrics over time.

    Parameters:
        plotter: Plotter instance for configuration access
        simulator: Simulator object with history
        metrics: List of metrics to display (default: cell counts and shear stress)
        save_path: Path to save the animation (default: auto-generated MP4 file)
        fps: Frames per second (default: 10)
        dpi: Dots per inch for the output file (default: 100)

    Returns:
        Animation object
    """
    # Default metrics if none provided
    if metrics is None:
        if 'healthy_cells' in simulator.history[0]:
            metrics = ['healthy_cells', 'senescent_tel', 'senescent_stress', 'input_value']
        else:
            metrics = ['cells', 'input_value']

    # Check if history exists
    if not simulator.history:
        print("No history data in simulator")
        return None

    # Extract time data
    time = np.array([state['time'] for state in simulator.history])

    # Convert time to hours if needed
    if plotter.config.time_unit == "minutes":
        time_in_hours = time / 60
        time_label = "Time (hours)"
    else:
        time_in_hours = time
        time_label = f"Time ({plotter.config.time_unit})"

    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3*len(metrics)), sharex=True)

    # Ensure axes is a list even for a single metric
    if len(metrics) == 1:
        axes = [axes]

    # Initialize line objects and metric data
    lines = []
    metric_data = {}

    for i, metric in enumerate(metrics):
        # Extract data for this metric
        if metric in simulator.history[0]:
            metric_data[metric] = np.array([state[metric] for state in simulator.history])

            # Create line
            line, = axes[i].plot([], [], 'r-', linewidth=2)
            lines.append(line)

            # Format axes
            axes[i].set_ylabel(metric, fontsize=12)
            axes[i].set_title(f'Evolution of {metric}', fontsize=12)
            axes[i].grid(True)

            # Set y-limits based on data range
            ymin = metric_data[metric].min() * 0.9
            ymax = metric_data[metric].max() * 1.1

            # Ensure non-zero range for y-axis
            if ymin == ymax:
                ymin -= 0.1
                ymax += 0.1

            axes[i].set_ylim(ymin, ymax)
        else:
            print(f"Metric '{metric}' not found in history data")

    # Set x-limits
    for ax in axes:
        ax.set_xlim(time_in_hours[0], time_in_hours[-1])

    # Add common x-label
    axes[-1].set_xlabel(time_label, fontsize=12)

    # Add title
    fig.suptitle('Endothelial Simulation Metrics Evolution', fontsize=14)
    plt.tight_layout()

    # Add time marker
    time_marker = plt.axvline(x=time_in_hours[0], color='k', linestyle='--')

    # Create animation update function
    def update(frame):
        # Update each line
        for i, metric in enumerate(metrics):
            if metric in metric_data:
                # Update data up to the current frame
                lines[i].set_data(time_in_hours[:frame+1], metric_data[metric][:frame+1])

        # Update time marker
        time_marker.set_xdata([time_in_hours[frame], time_in_hours[frame]])

        return lines + [time_marker]

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(simulator.history),
        blit=True, interval=1000/fps
    )

    # Generate save path if not provided
    if save_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(plotter.config.plot_directory, f"metrics_animation_{timestamp}.mp4")

    # Save animation
    print(f"Saving animation to {save_path}...")

    # Check if ffmpeg writer is available
    if 'ffmpeg' in animation.writers.list():
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Endothelial Simulation'), bitrate=1800)
        ani.save(save_path, writer=writer, dpi=dpi)
        print(f"Animation saved to {save_path}")
    else:
        print("Error: ffmpeg writer not available. Please install ffmpeg to save animations.")
        print("You can install ffmpeg using: pip install ffmpeg-python")

    return ani


def create_combined_animation(plotter, simulator, save_path=None, fps=10, dpi=100, frame_interval=5):
    """
    Create a combined animation showing cells and metrics.

    Parameters:
        plotter: Plotter instance for configuration access
        simulator: Simulator object after running the simulation
        save_path: Path to save the animation (default: auto-generated)
        fps: Frames per second (default: 10)
        dpi: Dots per inch for the output file (default: 100)
        frame_interval: Number of frames to skip between snapshots (default: 5)

    Returns:
        Animation object
    """
    # Check if history exists
    if not simulator.history:
        print("No history data in simulator")
        return None

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(12, 12))

    # Cell visualization subplot
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.set_xlim(0, simulator.grid.width)
    ax1.set_ylim(0, simulator.grid.height)
    ax1.set_xlabel('X Position (pixels)', fontsize=12)
    ax1.set_ylabel('Y Position (pixels)', fontsize=12)
    ax1.set_title('Endothelial Cell State', fontsize=14)

    # Create legend for cell visualization
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', alpha=0.7, label='Healthy'),
        Patch(facecolor='red', edgecolor='black', alpha=0.7, label='Telomere-Senescent'),
        Patch(facecolor='blue', edgecolor='black', alpha=0.7, label='Stress-Senescent')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # Add flow direction indicator
    arrow_length = simulator.grid.width * 0.1
    arrow_x = simulator.grid.width * 0.5
    arrow_y = simulator.grid.height * 0.05
    ax1.arrow(arrow_x - arrow_length / 2, arrow_y, arrow_length, 0,
             head_width=arrow_length * 0.2, head_length=arrow_length * 0.2,
             fc='black', ec='black', width=arrow_length * 0.05)
    ax1.text(arrow_x, arrow_y - arrow_length * 0.3, "Flow Direction",
            ha='center', va='top', fontsize=10)

    # Metrics subplot
    ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)

    # Extract time data
    time = np.array([state['time'] for state in simulator.history])

    # Convert time to hours if needed
    if plotter.config.time_unit == "minutes":
        time_in_hours = time / 60
        time_label = "Time (hours)"
    else:
        time_in_hours = time
        time_label = f"Time ({plotter.config.time_unit})"

    # Set up metrics plot
    ax2.set_xlim(time_in_hours[0], time_in_hours[-1])
    ax2.set_xlabel(time_label, fontsize=12)
    ax2.grid(True)

    # Determine metrics to show
    if 'healthy_cells' in simulator.history[0]:
        # Set up lines for cell counts
        healthy_line, = ax2.plot([], [], 'g-', linewidth=2, label='Healthy')
        sen_tel_line, = ax2.plot([], [], 'r-', linewidth=2, label='Tel-Senescent')
        sen_stress_line, = ax2.plot([], [], 'b-', linewidth=2, label='Stress-Senescent')

        # Extract data
        healthy_data = np.array([state['healthy_cells'] for state in simulator.history])
        sen_tel_data = np.array([state['senescent_tel'] for state in simulator.history])
        sen_stress_data = np.array([state['senescent_stress'] for state in simulator.history])

        # Set y-limits based on data
        max_cells = max(np.max(healthy_data), np.max(sen_tel_data), np.max(sen_stress_data)) * 1.1
        ax2.set_ylim(0, max_cells)
        ax2.set_ylabel('Cell Count', fontsize=12)
        ax2.set_title('Cell Population Dynamics', fontsize=12)
        ax2.legend(loc='upper left')

        metrics_lines = [healthy_line, sen_tel_line, sen_stress_line]
        metrics_data = [healthy_data, sen_tel_data, sen_stress_data]
    else:
        # Simpler case with only total cells
        cells_line, = ax2.plot([], [], 'k-', linewidth=2, label='Total Cells')

        # Extract data
        cells_data = np.array([state['cells'] for state in simulator.history])

        # Set y-limits
        max_cells = np.max(cells_data) * 1.1
        ax2.set_ylim(0, max_cells)
        ax2.set_ylabel('Cell Count', fontsize=12)
        ax2.set_title('Cell Population', fontsize=12)
        ax2.legend(loc='upper left')

        metrics_lines = [cells_line]
        metrics_data = [cells_data]

    # Add time marker
    time_marker = ax2.axvline(x=time_in_hours[0], color='k', linestyle='--')

    # Add time text
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Create empty container for ellipses
    ellipses = []

    # Select frames to use (to keep animation size manageable)
    num_frames = len(simulator.history)
    selected_frames = list(range(0, num_frames, frame_interval))
    if selected_frames[-1] != num_frames - 1:
        selected_frames.append(num_frames - 1)  # Include the last frame

    # Update function for animation
    def update(frame_idx):
        # Get frame from history based on index
        frame = selected_frames[min(frame_idx, len(selected_frames) - 1)]
        state = simulator.history[frame]

        # Clear previous ellipses
        for ellipse in ellipses:
            ellipse.remove()
        ellipses.clear()

        # Update metrics lines
        for i, line in enumerate(metrics_lines):
            line.set_data(time_in_hours[:frame+1], metrics_data[i][:frame+1])

        # Update time marker
        time_marker.set_xdata([time_in_hours[frame], time_in_hours[frame]])

        # Display cells based on current state
        # Note: Since we don't have historical cell positions,
        # we use the final cell positions but update colors and counts
        # to reflect the state at this time point

        # Get cell counts
        if 'healthy_cells' in state:
            healthy_count = state['healthy_cells']
            sen_tel_count = state['senescent_tel']
            sen_stress_count = state['senescent_stress']
        else:
            total_count = state['cells']

        # Create a visualization based on current simulation state
        # This is a simplification since we don't have cell positions over time
        # Instead, we'll show the current cells with appropriate counts

        # We'll visualize cells as a grid for demonstration
        # (since we don't have historical positions)
        grid_size = int(np.sqrt(len(simulator.grid.cells))) + 1
        cell_size = min(simulator.grid.width, simulator.grid.height) / (grid_size + 1)

        cell_idx = 0

        # First, count how many of each type to show
        if 'healthy_cells' in state:
            total_cells = healthy_count + sen_tel_count + sen_stress_count
        else:
            total_cells = total_count

        # Current cells as a baseline for visualization
        current_cells = list(simulator.grid.cells.values())
        num_current_cells = len(current_cells)

        # Calculate scaling ratio to estimate historical cells
        scale_ratio = total_cells / max(1, num_current_cells)

        # Create a grid layout for cells
        cells_to_show = min(100, int(total_cells))  # Limit to 100 cells for performance

        for i in range(cells_to_show):
            # Calculate grid position
            row = i // grid_size
            col = i % grid_size

            x = cell_size * (col + 1)
            y = cell_size * (row + 1)

            # Sample from current cells for properties (repeating if needed)
            cell_template = current_cells[i % num_current_cells]

            # Get cell properties from template
            orientation = cell_template.orientation
            aspect_ratio = cell_template.aspect_ratio
            area = cell_template.area

            # Determine cell type based on counts
            if 'healthy_cells' in state:
                # Assign cell type based on proportions
                if i < healthy_count * cells_to_show / total_cells:
                    is_senescent = False
                    senescence_cause = None
                    color = 'green'
                elif i < (healthy_count + sen_tel_count) * cells_to_show / total_cells:
                    is_senescent = True
                    senescence_cause = 'telomere'
                    color = 'red'
                else:
                    is_senescent = True
                    senescence_cause = 'stress'
                    color = 'blue'
            else:
                # Default to healthy
                is_senescent = False
                color = 'green'

            # Calculate ellipse dimensions
            a = np.sqrt(area * aspect_ratio) * 0.8  # Semi-major axis, scaled down to avoid overlap
            b = area / a * 0.8  # Semi-minor axis, scaled down

            # Create ellipse
            ellipse = Ellipse(
                xy=(x, y),
                width=2 * a,
                height=2 * b,
                angle=np.degrees(orientation),
                alpha=0.7
            )

            ellipse.set_facecolor(color)
            ellipse.set_edgecolor('black')
            ellipse.set_linewidth(0.5)

            # Add to plot
            ellipses.append(ellipse)
            ax1.add_patch(ellipse)

        # Update info text
        time_val = state['time']
        if plotter.config.time_unit == "minutes":
            time_str = f"{time_val:.1f} min ({time_val/60:.1f} hours)"
        else:
            time_str = f"{time_val:.1f} {plotter.config.time_unit}"

        if 'healthy_cells' in state:
            info_text = (
                f"Time: {time_str}\n"
                f"Shear Stress: {state['input_value']:.2f} Pa\n"
                f"Healthy Cells: {state['healthy_cells']}\n"
                f"Tel-Senescent: {state['senescent_tel']}\n"
                f"Stress-Senescent: {state['senescent_stress']}"
            )
        else:
            info_text = (
                f"Time: {time_str}\n"
                f"Shear Stress: {state['input_value']:.2f} Pa\n"
                f"Total Cells: {state['cells']}"
            )

        time_text.set_text(info_text)

        return ellipses + metrics_lines + [time_marker, time_text]

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(selected_frames),
        blit=True, interval=1000/fps
    )

    # Generate save path if not provided
    if save_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(plotter.config.plot_directory, f"cell_animation_{timestamp}.mp4")

    # Save animation
    print(f"Saving animation to {save_path}...")

    # Check if ffmpeg writer is available
    if 'ffmpeg' in animation.writers.list():
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Endothelial Simulation'), bitrate=1800)
        ani.save(save_path, writer=writer, dpi=dpi)
        print(f"Animation saved to {save_path}")
    else:
        print("Error: ffmpeg writer not available. Please install ffmpeg to save animations.")
        print("You can install ffmpeg using: pip install ffmpeg-python")

    plt.tight_layout()
    return ani


def record_cell_states(plotter, simulator, record_interval=10):
    """
    Set up recording of cell states during simulation for animation.
    This function returns a list object that will be populated with
    cell states during the simulation run.

    Parameters:
        plotter: Plotter instance for configuration access
        simulator: Simulator object before running
        record_interval: Number of simulation steps between recorded frames

    Returns:
        frame_data: List that will be populated with cell states during simulation
    """
    # Create container for frame data
    frame_data = []

    # Store original step method
    original_step = simulator.step

    # Define wrapper for step method to record frames
    def step_with_recording():
        # Call original step method
        result = original_step()

        # Record frame at specified intervals
        if simulator.step_count % record_interval == 0:
            # Collect cell data for this frame
            cells_data = []

            for cell_id, cell in simulator.grid.cells.items():
                cells_data.append({
                    'cell_id': cell_id,
                    'position': cell.position,
                    'orientation': cell.orientation,
                    'aspect_ratio': cell.aspect_ratio,
                    'area': cell.area,
                    'is_senescent': cell.is_senescent,
                    'senescence_cause': cell.senescence_cause
                })

            # Store frame data
            frame_data.append({
                'time': simulator.time,
                'input_value': simulator.input_pattern['value'],
                'cell_count': len(simulator.grid.cells),
                'cells': cells_data
            })

        return result

    # Replace step method with wrapped version
    simulator.step = step_with_recording
    simulator._original_step = original_step

    return frame_data


def create_detailed_cell_animation(plotter, frame_data, simulator, save_path=None, fps=10, dpi=100):
    """
    Create a detailed animation from recorded cell states.

    Parameters:
        plotter: Plotter instance for configuration access
        frame_data: List of recorded frame data (from record_cell_states)
        simulator: Simulator object
        save_path: Path to save the animation (default: auto-generated MP4 file)
        fps: Frames per second (default: 10)
        dpi: Dots per inch for the output file (default: 100)

    Returns:
        Animation object
    """
    # Check if frame data exists
    if not frame_data:
        print("No frame data available")
        return None

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set axis limits based on grid size
    ax.set_xlim(0, simulator.grid.width)
    ax.set_ylim(0, simulator.grid.height)

    # Create empty container for time-varying texts
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Create placeholder for legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', alpha=0.7, label='Healthy'),
        Patch(facecolor='red', edgecolor='black', alpha=0.7, label='Telomere-Senescent'),
        Patch(facecolor='blue', edgecolor='black', alpha=0.7, label='Stress-Senescent')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add flow direction indicator
    arrow_length = simulator.grid.width * 0.1
    arrow_x = simulator.grid.width * 0.5
    arrow_y = simulator.grid.height * 0.05

    ax.arrow(arrow_x - arrow_length / 2, arrow_y, arrow_length, 0,
             head_width=arrow_length * 0.2, head_length=arrow_length * 0.2,
             fc='black', ec='black', width=arrow_length * 0.05)

    ax.text(arrow_x, arrow_y - arrow_length * 0.3, "Flow Direction",
            ha='center', va='top', fontsize=10)

    # Add title
    ax.set_title('Endothelial Cell Evolution', fontsize=14)
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)

    # Create empty container for ellipses
    ellipses = []

    # Create animation update function
    def update(frame):
        # Clear previous ellipses
        for ellipse in ellipses:
            ellipse.remove()
        ellipses.clear()

        # Get frame data
        data = frame_data[min(frame, len(frame_data) - 1)]

        # Plot cells for this frame
        for cell_data in data['cells']:
            # Get cell properties
            x, y = cell_data['position']
            orientation = cell_data['orientation']
            aspect_ratio = cell_data['aspect_ratio']
            area = cell_data['area']
            is_senescent = cell_data['is_senescent']
            senescence_cause = cell_data['senescence_cause']

            # Calculate ellipse dimensions
            a = np.sqrt(area * aspect_ratio)  # Semi-major axis
            b = area / a  # Semi-minor axis

            # Create ellipse
            ellipse = Ellipse(
                xy=(x, y),
                width=2 * a,
                height=2 * b,
                angle=np.degrees(orientation),
                alpha=0.7
            )

            # Set color based on cell type
            if not is_senescent:
                color = 'green'
            elif senescence_cause == 'telomere':
                color = 'red'
            else:  # stress-induced
                color = 'blue'

            ellipse.set_facecolor(color)
            ellipse.set_edgecolor('black')
            ellipse.set_linewidth(0.5)

            # Add to plot
            ellipses.append(ellipse)
            ax.add_patch(ellipse)

        # Update info text
        time_val = data['time']
        if plotter.config.time_unit == "minutes":
            time_str = f"{time_val:.1f} min ({time_val/60:.1f} hours)"
        else:
            time_str = f"{time_val:.1f} {plotter.config.time_unit}"

        info_text = (
            f"Time: {time_str}\n"
            f"Shear Stress: {data['input_value']:.2f} Pa\n"
            f"Cells: {data['cell_count']}"
        )
        time_text.set_text(info_text)

        return ellipses + [time_text]

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_data),
        blit=True, interval=1000/fps
    )

    # Generate save path if not provided
    if save_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(plotter.config.plot_directory, f"detailed_animation_{timestamp}.mp4")

    # Save animation
    print(f"Saving animation to {save_path}...")

    # Check if ffmpeg writer is available
    if 'ffmpeg' in animation.writers.list():
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Endothelial Simulation'), bitrate=1800)
        ani.save(save_path, writer=writer, dpi=dpi)
        print(f"Animation saved to {save_path}")
    else:
        print("Error: ffmpeg writer not available. Please install ffmpeg to save animations.")
        print("You can install ffmpeg using: pip install ffmpeg-python")

    return ani