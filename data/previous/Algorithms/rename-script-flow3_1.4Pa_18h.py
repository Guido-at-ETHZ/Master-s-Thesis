import os
import re
from datetime import datetime


def parse_original_filename(filename):
    """Parse information from original filename format."""
    # Examples: 
    # flow_3_1.4_L2R_20x_trans_001.nd2
    # flow_3_1.4_L2R_20x_topo_001.nd2
    # flow_3_1.4_L2R_20x_flat_001.nd2
    # flow_3_static_20x_flat_001.nd2

    # Patterns for flow3_1.4Pa_18h files
    patterns = [
        # Pattern for detailed filenames (with imaging type)
        r'flow_3_(?:1\.4_L2R|static)_(\d+x)_(\w+)_(\d+).nd2',
        # Pattern for simple static files
        r'flow_3_static_(\d+x)_(\d+).nd2'
    ]
    match = None
    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            break

    if not match:
        return None

    if len(match.groups()) == 3:
        # Detailed filename with imaging type
        magnification, imaging_type, sequence = match.groups()
    else:
        # Simple static filename
        magnification, sequence = match.groups()
        imaging_type = "topoA"  # Default to TopoA for files without explicit imaging type

    # Determine pressure and flow direction from filename
    is_static = 'static' in filename

    return {
        'pressure': '0Pa' if is_static else '1.4Pa',
        'series': 'U',  # Unspecified series
        'date': '05mar19',  # Hardcoded date as specified
        'magnification': magnification,
        'flow_direction': 'L2R' if not is_static else 'L2RA',
        'imaging_type': imaging_type.capitalize(),  # Capitalize first letter
        'sequence': f"seq{sequence.zfill(3)}"
    }


def generate_new_filename(components):
    """Generate new filename based on components."""
    if not components:
        return None

    template = "{pressure}_{series}_{date}_{magnification}_{flow_direction}_{imaging_type}_{sequence}.nd2"
    return template.format(**components)


def rename_files_in_directory(directory_path):
    """Rename all files in the specified directory."""
    successfully_renamed = []
    failed_to_rename = []

    # Get list of .nd2 files in directory
    files = [f for f in os.listdir(directory_path) if f.endswith('.nd2')]

    for original_filename in files:
        # Parse components from original filename
        components = parse_original_filename(original_filename)

        if components:
            # Generate new filename
            new_filename = generate_new_filename(components)

            if new_filename:
                # Full paths for rename operation
                old_path = os.path.join(directory_path, original_filename)
                new_path = os.path.join(directory_path, new_filename)

                try:
                    print(f"Renaming: {old_path} -> {new_path}")
                    os.rename(old_path, new_path)  # Uncomment this line to perform actual renaming
                    successfully_renamed.append((original_filename, new_filename))
                    print(f"Successfully renamed {original_filename} to {new_filename}")
                except Exception as e:
                    failed_to_rename.append((original_filename, str(e)))
                    print(f"Error renaming {original_filename}: {str(e)}")
            else:
                failed_to_rename.append((original_filename, "Failed to generate new filename"))
        else:
            failed_to_rename.append((original_filename, "Failed to parse filename components"))

    return successfully_renamed, failed_to_rename


# Test the script
if __name__ == "__main__":
    # Test with flow3_1.4Pa_18h folder
    directory = "flow3_1.4Pa_18h"
    successfully_renamed, failed_to_rename = rename_files_in_directory(directory)

    print("\nSuccessfully parsed and ready to rename:")
    for old, new in successfully_renamed:
        print(f"{old} -> {new}")

    print("\nFailed to rename:")
    for filename, error in failed_to_rename:
        print(f"{filename}: {error}")