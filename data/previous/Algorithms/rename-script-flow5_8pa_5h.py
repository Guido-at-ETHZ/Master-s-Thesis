import os
import re
from datetime import datetime

def parse_original_filename(filename):
    """Parse information from original filename format."""
    # Examples: 
    # flow5_8Pa_to_1.4Pa_R2L_flat_40x004.nd2
    # flow5_8Pa_to_1.4Pa_R2L_topo_20x006.nd2
    # flow5_static_flat_20x001.nd2
    
    patterns = [
        # Pattern for pressure transition files
        r'flow5_8Pa_to_1\.4Pa_R2L_(\w+)_(\d+x)(\d+).nd2',
        # Pattern for static files
        r'flow5_static_(\w+)_(\d+x)(\d+).nd2',
        # Pattern for static R2L files
        r'flow5_static_R2L_(\w+)_(\d+x)(\d+).nd2'
    ]
    
    match = None
    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            break
    
    if not match:
        return None
    
    imaging_type, magnification, sequence = match.groups()
    
    # Determine pressure from filename
    if 'static' in filename:
        pressure = '0Pa'
    else:
        pressure = '8Pa-1.4Pa'  # Indicating pressure transition
    
    # Determine flow direction
    if 'R2L' in filename:
        flow_direction = 'R2L'
    else:
        flow_direction = 'L2RA'  # Default for static without R2L
    
    return {
        'pressure': pressure,
        'series': 'U',          # Unspecified series
        'date': '03apr19',      # Hardcoded date as specified
        'magnification': magnification,
        'flow_direction': flow_direction,
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
    # Test with flow5_8pa_5h folder
    directory = "flow5_8pa_5h"
    successfully_renamed, failed_to_rename = rename_files_in_directory(directory)

    print("\nSuccessfully parsed and ready to rename:")
    for old, new in successfully_renamed:
        print(f"{old} -> {new}")

    print("\nFailed to rename:")
    for filename, error in failed_to_rename:
        print(f"{filename}: {error}")
