# Stress Factor Parameter Calibration
import numpy as np

# System parameters
cellular_resistance = 0.5  # From config.base_cellular_resistance
time_18h = 18.0  # Target time period (hours)

print("🔧 STRESS FACTOR CALIBRATION")
print("=" * 50)
print(f"Cellular resistance threshold: {cellular_resistance}")
print(f"Target time period: {time_18h} hours")

# STEP 1: Calibrate baseline stress factor (at 0 Pa)
print(f"\n📊 STEP 1: BASELINE STRESS FACTOR (0 Pa)")
print("Goal: Minimal stress-induced senescence at 0 Pa")

# For 5% total senescence at 0 Pa, we want most to be telomere-mediated
# Individual stress senescence should be < 1% contribution
target_stress_contribution = 0.01  # 1% of population
target_stress_ratio_18h = 0.6  # 60% of resistance threshold after 18h

baseline_stress_factor = (target_stress_ratio_18h * cellular_resistance) / time_18h
print(f"Target stress ratio after 18h: {target_stress_ratio_18h:.1%}")
print(f"Calculated baseline stress factor: {baseline_stress_factor:.4f}")

# Compare with your current value
your_baseline = 0.002778
print(f"Your current baseline: {your_baseline:.6f}")
print(f"Your 18h stress ratio: {(your_baseline * time_18h / cellular_resistance):.1%}")

# STEP 2: Calibrate stress sensitivity (tau coefficient)
print(f"\n📊 STEP 2: STRESS SENSITIVITY COEFFICIENT")
print("Goal: Define shear stress levels that cause senescence")

# Target scenarios for different shear stress levels
stress_scenarios = [
    {"shear": 5, "time": 24, "senescence_target": 0.1},  # 10% at 5 Pa in 24h
    {"shear": 10, "time": 18, "senescence_target": 0.5},  # 50% at 10 Pa in 18h
    {"shear": 15, "time": 12, "senescence_target": 0.8},  # 80% at 15 Pa in 12h
    {"shear": 20, "time": 6, "senescence_target": 0.9},  # 90% at 20 Pa in 6h
]

print("Target senescence scenarios:")
for scenario in stress_scenarios:
    tau = scenario["shear"]
    time_h = scenario["time"]
    target_ratio = scenario["senescence_target"]

    # Required stress factor to achieve target
    required_stress_factor = (target_ratio * cellular_resistance) / time_h

    # Calculate required sensitivity coefficient
    # stress_factor = baseline + tau * sensitivity
    # required_stress_factor = baseline + tau * sensitivity
    # sensitivity = (required_stress_factor - baseline) / tau
    required_sensitivity = (required_stress_factor - baseline_stress_factor) / tau

    print(f"  {tau:2d} Pa, {time_h:2d}h → {target_ratio:.0%} senescence")
    print(f"     Required stress factor: {required_stress_factor:.4f}")
    print(f"     Required sensitivity: {required_sensitivity:.6f}")

# STEP 3: Choose optimal parameters
print(f"\n📊 STEP 3: PARAMETER RECOMMENDATIONS")

# Use moderate scenario (10 Pa, 18h, 50% senescence) as calibration point
calibration_scenario = stress_scenarios[1]
tau_cal = calibration_scenario["shear"]
time_cal = calibration_scenario["time"]
target_cal = calibration_scenario["senescence_target"]

required_stress_factor_cal = (target_cal * cellular_resistance) / time_cal
optimal_sensitivity = (required_stress_factor_cal - baseline_stress_factor) / tau_cal

print(f"Calibration point: {tau_cal} Pa, {time_cal}h, {target_cal:.0%} senescence")
print(f"Optimal parameters:")
print(f"  baseline_stress_factor = {baseline_stress_factor:.6f}")
print(f"  stress_sensitivity = {optimal_sensitivity:.6f}")

# STEP 4: Validate with all scenarios
print(f"\n📊 STEP 4: VALIDATION")
print("Predicted senescence with optimal parameters:")

for scenario in stress_scenarios:
    tau = scenario["shear"]
    time_h = scenario["time"]
    target = scenario["senescence_target"]

    # Calculate stress factor with optimal parameters
    stress_factor = baseline_stress_factor + tau * optimal_sensitivity
    final_stress = stress_factor * time_h
    stress_ratio = final_stress / cellular_resistance

    print(f"  {tau:2d} Pa, {time_h:2d}h → {stress_ratio:.0%} senescence (target: {target:.0%})")

# STEP 5: Code implementation
print(f"\n💻 CODE IMPLEMENTATION:")
print("def _calculate_stress_factor(self, config):")
print('    """Calculate stress factor based on shear stress magnitude."""')
print("    tau = self.local_shear_stress")
print("    if tau <= 0:")
print(f"        return {baseline_stress_factor:.6f}")
print("    else:")
print(f"        return {baseline_stress_factor:.6f} + tau * {optimal_sensitivity:.6f}")

# STEP 6: Compare with your current parameters
print(f"\n🔍 COMPARISON WITH YOUR CURRENT PARAMETERS:")
your_sensitivity = 0.0009921

scenarios_comparison = [0, 5, 10, 15, 20]  # Pa
print("Shear | Your Result | Optimal Result | 18h Senescence %")
print("------|-------------|----------------|------------------")
for tau in scenarios_comparison:
    # Your current parameters
    your_factor = your_baseline + tau * your_sensitivity if tau > 0 else your_baseline
    your_18h_ratio = (your_factor * 18) / cellular_resistance

    # Optimal parameters
    opt_factor = baseline_stress_factor + tau * optimal_sensitivity if tau > 0 else baseline_stress_factor
    opt_18h_ratio = (opt_factor * 18) / cellular_resistance

    print(f"{tau:4d}  | {your_18h_ratio:10.0%} | {opt_18h_ratio:13.0%} | Target varies")

# STEP 7: Fine-tuning recommendations
print(f"\n🎯 FINE-TUNING RECOMMENDATIONS:")
print("1. Start with optimal parameters")
print("2. Run simulation at 0 Pa for 18h - should get ~5% senescence total")
print("3. If stress contribution > 1%, reduce baseline_stress_factor")
print("4. Test at 10 Pa for 18h - should get ~50% stress senescence")
print("5. Adjust stress_sensitivity if needed")

print(f"\n⚡ FINAL PARAMETERS:")
print(f"baseline_stress_factor = {baseline_stress_factor:.6f}  # {target_stress_ratio_18h:.0%} resistance at 18h")
print(f"stress_sensitivity = {optimal_sensitivity:.6f}     # 50% senescence at 10 Pa, 18h")