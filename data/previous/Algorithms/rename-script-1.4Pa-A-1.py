import os
import re
from datetime import datetime

def parse_original_filename(filename):
    """Parse information from original filename format."""
    # Examples: 
    # H_P3-2-1.4Pa_A_70c-30T_19-20.12.21_40x-001.nd2
    # H_P3-2-1.4Pa_A_70c-30T_19-20.12.21-001.nd2
    
    # Pattern for 1.4Pa-A-1
    pattern = r'H_P3-2-1\.4Pa_A_70c-30T_(\d{2}-\d{2}.\d{2}.\d{2})(?:_(\d+x))?-?(\d+).nd2'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    date_str, magnification, sequence = match.groups()
    
    # Convert date format
    try:
        # Parse date from "19-20.12.21" format - using first date
        date = datetime.strptime(date_str.split('-')[0] + "." + date_str.split('.', 1)[1], "%d.%m.%y")
        formatted_date = date.strftime("%d%b%y").lower()
    except:
        formatted_date = "UnknownDate"
    
    return {
        'pressure': '1.4Pa',
        'series': 'A1',     # First experiment
        'date': formatted_date,
        'magnification': magnification if magnification else '20xA',  # Default to 20xA if not specified
        'flow_direction': 'L2RA',  # default
        'imaging_type': 'FlatA',   # default
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
                    #os.rename(old_path, new_path)  # Uncomment this line to perform actual renaming
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
    # Test with 1.4Pa-A-1 folder
    directory = "1.4Pa-A-2_40x"
    successfully_renamed, failed_to_rename = rename_files_in_directory(directory)

    print("\nSuccessfully parsed and ready to rename:")
    for old, new in successfully_renamed:
        print(f"{old} -> {new}")

    print("\nFailed to rename:")
    for filename, error in failed_to_rename:
        print(f"{filename}: {error}")
