# test_distributions.py
from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.core import Simulator
from endothelial_simulation.visualization import Plotter

# Create a short simulation for testing
config = SimulationConfig()
config.simulation_duration = 240  # 4 hours (short test)
config.initial_cell_count = 50   # Fewer cells for faster testing

# Run simulation
simulator = Simulator(config)
simulator.initialize()
simulator.set_constant_input(1.4)  # 1.4 Pa shear stress
print("Running test simulation...")
simulator.run()

# Test the new plotting function
plotter = Plotter(config)
print("Creating hourly distribution plots...")
fig = plotter.plot_hourly_cell_distributions(simulator, max_hours=4)

if fig:
    print("✅ Success! Check your results/ folder for the plot.")
else:
    print("❌ Something went wrong.")