# Endothelial Cell Mechanotransduction Simulation

This repository contains a computational model for simulating endothelial cell mechanotransduction in response to varying mechanical stimuli, with a focus on shear stress. The simulation integrates multiple components including temporal dynamics, population dynamics, and spatial properties to provide a comprehensive framework for studying endothelial cell behavior.

## Features

- **Temporal Dynamics**: Models how cell responses evolve over time after mechanical stimulation
- **Population Dynamics**: Tracks cell proliferation, senescence, and death under different conditions
- **Spatial Properties**: Simulates morphological adaptations such as orientation and aspect ratio
- **Visualization Tools**: Comprehensive plotting capabilities for analyzing simulation results
- **Parameter Study**: Tools for exploring parameter space and identifying optimal conditions

## Project Structure

```
endothelial_simulation/
├── core/                      # Core simulation infrastructure
│   ├── __init__.py
│   ├── cell.py                # Cell class for individual cell properties
│   ├── grid.py                # Spatial grid for cell arrangement
│   └── simulator.py           # Main simulation engine
├── models/                    # Model components
│   ├── __init__.py
│   ├── temporal_dynamics.py   # Temporal adaptation model
│   ├── population_dynamics.py # Cell population evolution model
│   └── spatial_properties.py  # Spatial characteristics model
├── visualization/             # Visualization tools
│   ├── __init__.py
│   └── plotters.py            # Plotting functions
├── results/                   # Default directory for simulation outputs
├── config.py                  # Configuration settings
├── main.py                    # Main script for running simulations
└── parameter_study.py         # Script for parameter exploration
```

## Usage

### Basic Simulation

To run a basic simulation with default parameters:

```bash
python main.py
```

### Customizing Input Parameters

You can customize the simulation parameters through command-line arguments:

```bash
python main.py --mode full --input-type step --initial-value 0 --final-value 45 --step-time 120 --duration 360
```

Available input types:
- `constant`: Constant shear stress
- `step`: Step change in shear stress
- `ramp`: Linear ramp in shear stress
- `oscillatory`: Oscillating shear stress

### Running a Parameter Study

To explore the effects of different parameter combinations:

```bash
python parameter_study.py
```

This will run simulations across multiple shear stress values, senolytic concentrations, and stem cell input rates, generating comprehensive analysis plots.

## Configuration

The simulation can be configured to focus on specific aspects:

- **Full Mode**: All components enabled (population dynamics, spatial properties, and temporal dynamics)
- **Temporal Mode**: Focus only on temporal dynamics
- **Spatial Mode**: Focus only on spatial properties
- **Minimal Mode**: Basic population dynamics without senescence

## Models

### Temporal Dynamics Model

The temporal dynamics model implements a first-order differential equation to describe cellular adaptation to mechanical stimuli:

```
dy/dt = (Amax(P) - y) / τ
```

Where:
- `y` is the cellular response
- `Amax(P)` is the maximum response at pressure P
- `τ` is the time constant that scales with Amax

### Population Dynamics Model

The population dynamics model tracks cells at different division stages and includes:

- Proliferation with density-dependent inhibition
- Age-dependent death rates
- Telomere-induced and stress-induced senescence
- Senolytic effects for targeted senescent cell removal
- Stem cell input for tissue renewal

### Spatial Properties Model

The spatial properties model simulates how cell morphology adapts to mechanical forces, including:

- Orientation relative to flow direction
- Aspect ratio (elongation)
- Cell area and shape index
- Alignment and confluency metrics

## Example Results

The simulation produces various visualizations to analyze cellular responses:

- Cell population dynamics over time
- Spatial metrics (alignment, shape index, confluency)
- Cell visualization with color-coded cell types
- Parameter study plots showing relationships between variables

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Pandas
- Multiprocessing (for parallel parameter studies)
- ffmpeg-python>=0.2.0