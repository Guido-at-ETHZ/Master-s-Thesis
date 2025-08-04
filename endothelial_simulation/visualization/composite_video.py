

import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, ImageSequenceClip, clips_array
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import glob

def create_composite_video(results_dir, simulation_data, mpc_data):
    """
    Generates a composite video with multiple synchronized panels.
    This version includes robust error checking, logging, and data validation.
    """
    print("\n" + "="*50)
    print("🎬 STARTING COMPOSITE VIDEO GENERATION")
    print("="*50)

    # 1. VALIDATE INPUTS
    print(f"🔍 Step 1: Validating inputs...")
    if not os.path.isdir(results_dir):
        print(f"❌ FATAL: Results directory not found at '{results_dir}'")
        return

    if not simulation_data:
        print("❌ FATAL: 'simulation_data' is empty. Cannot proceed.")
        return

    if not mpc_data:
        print("❌ FATAL: 'mpc_data' is empty. Cannot proceed.")
        return
    print("   ✅ Inputs validated.")

    # 2. FIND AND LOAD THE MAIN ANIMATION
    print(f"🔍 Step 2: Locating main animation video...")
    mosaic_animation_pattern = os.path.join(results_dir, "REAL_mosaic_animation_*.mp4")
    animation_files = glob.glob(mosaic_animation_pattern)
    if not animation_files:
        print(f"❌ FATAL: No mosaic animation file found matching '{mosaic_animation_pattern}'")
        return

    mosaic_animation_path = max(animation_files, key=os.path.getctime)
    print(f"   ✅ Found animation: {os.path.basename(mosaic_animation_path)}")

    try:
        main_clip = VideoFileClip(mosaic_animation_path)
        fps = main_clip.fps
        duration = main_clip.duration
        print(f"   ✅ Video loaded successfully (Duration: {duration:.2f}s, FPS: {fps})")
    except Exception as e:
        print(f"❌ FATAL: Failed to load video with moviepy: {e}")
        return

    # 3. EXTRACT AND PREPARE DATA
    print("🔍 Step 3: Extracting and preparing data for plots...")
    try:
        sim_times = np.array([d['time'] for d in simulation_data])
        orientations = [d.get('cell_properties', {}).get('orientations', []) for d in simulation_data]
        alignments = [d.get('alignment_index', 0) for d in simulation_data]

        mpc_times = np.array([d['time'] for d in mpc_data])
        shear_stresses = [d['shear_stress'] for d in mpc_data]
        mpc_targets = [d['target'] for d in mpc_data]
        mpc_actuals = [d['actual'] for d in mpc_data]
        print(f"   ✅ Data extracted (Sim points: {len(sim_times)}, MPC points: {len(mpc_times)})")
    except KeyError as e:
        print(f"❌ FATAL: Missing expected key in data structure - {e}")
        return

    # 4. GENERATE FRAMES FOR DYNAMIC PLOTS
    total_frames = int(duration * fps)
    print(f"🎬 Step 4: Generating {total_frames} frames for dynamic plots...")
    dynamic_frames = []

    # Correctly map video time to simulation time
    total_sim_time = sim_times[-1]
    print(f"   Mapping video duration ({duration:.2f}s) to simulation duration ({total_sim_time:.2f} minutes).")

    for frame_num in range(total_frames):
        # Proportional time mapping
        t_sim_minutes = (frame_num / total_frames) * total_sim_time

        # Find the closest data indices
        sim_idx = np.argmin(np.abs(sim_times - t_sim_minutes))
        mpc_idx = np.argmin(np.abs(mpc_times - t_sim_minutes))

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), dpi=120)
        fig.suptitle(f"Time: {t_sim_minutes:.1f} min", fontsize=14)

        # a) MPC Input Panel
        ax1 = axes[0]
        ax1.plot(mpc_times[:mpc_idx+1], shear_stresses[:mpc_idx+1], label='Shear Stress (Pa)', color='red')
        ax1.plot(mpc_times[:mpc_idx+1], mpc_targets[:mpc_idx+1], label='MPC Target', linestyle='--', color='blue')
        ax1.plot(mpc_times[:mpc_idx+1], mpc_actuals[:mpc_idx+1], label='Actual Response', color='green')
        ax1.set_title("MPC Control Inputs")
        ax1.set_ylabel("Value")
        ax1.set_xlim(0, max(sim_times))
        ax1.legend(fontsize='small')
        ax1.grid(True)

        # b) Angle Distribution Panel
        ax2 = axes[1]
        if orientations[sim_idx]:
            ax2.hist(orientations[sim_idx], bins=20, range=(0, 90), color='purple', alpha=0.7)
        ax2.set_title("Cell Orientation Angles")
        ax2.set_xlabel("Angle (degrees from flow)")
        ax2.set_ylabel("Frequency")
        ax2.set_xlim(0, 90)

        # c) Flow Alignment Panel
        ax3 = axes[2]
        ax3.plot(sim_times[:sim_idx+1], alignments[:sim_idx+1], color='orange')
        ax3.set_title("Flow Alignment Index")
        ax3.set_xlabel("Time (minutes)")
        ax3.set_ylabel("Alignment Index")
        ax3.set_xlim(0, max(sim_times))
        ax3.set_ylim(0, 1)
        ax3.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        canvas = FigureCanvas(fig)
        canvas.draw()
        # Use the modern method to get the buffer as a NumPy array
        buf = canvas.buffer_rgba()
        frame_image = np.asarray(buf)
        # Moviepy expects RGB, so we slice off the Alpha channel
        dynamic_frames.append(frame_image[:, :, :3])
        plt.close(fig)

        if frame_num % int(fps * 2) == 0: # Log progress every 2 seconds of video
            print(f"   ...Generated frame {frame_num}/{total_frames}")

    print("   ✅ All dynamic frames generated.")

    # 5. ASSEMBLE THE FINAL VIDEO
    print("🎞️  Step 5: Assembling final video...")
    try:
        dynamic_clip = ImageSequenceClip(dynamic_frames, fps=fps)

        # Resize clips to have the same height for side-by-side composition
        final_height = 1080
        main_resized = main_clip.resize(height=final_height)
        dynamic_resized = dynamic_clip.resize(height=final_height)

        composite_clip = clips_array([[main_resized, dynamic_resized]])

        output_path = os.path.join(results_dir, "composite_video_comparison.mp4")
        composite_clip.write_videofile(output_path, fps=fps, codec='libx264')
        print(f"   ✅ Composite video successfully saved to: {output_path}")

    except Exception as e:
        print(f"❌ FATAL: Failed during final video assembly with moviepy: {e}")

    print("="*50)
    print("🏁 COMPOSITE VIDEO GENERATION FINISHED")
    print("="*50 + "\n")
