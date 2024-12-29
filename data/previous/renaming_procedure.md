# Dataset Organization and File Naming Convention

## Overview
This repository contains microscopy data of endothelial cells under various flow conditions. The data is organized using a systematic file naming convention that encodes experimental conditions and imaging parameters.

## File Naming Convention
Files follow this structure:
```
[Pressure]_[Series]_[Date]_[Magnification]_[FlowDirection]_[ImagingType]_[Sequence].nd2
```

### Parameter Descriptions

#### Pressure
Indicates the shear stress applied to the cells:
- `0Pa`: Static condition (no flow)
- `1.4Pa`: Constant flow at 1.4 Pascal
- `8Pa-1.4Pa`: Pressure transition from 8Pa to 1.4Pa

#### Series
Indicates the experimental series and cell treatment:
- `A1`: 70% control + 30% TNF-α treated cells
- `U`: Unspecified/standard HUVEC cells

#### Date
Format: DDmmmYY (lowercase)
- Example: `19dec21`, `05mar19`, `03apr19`

#### Magnification
Microscope objective used:
- `20x`: 20x magnification
- `40x`: 40x magnification
- `20xA`: 20x magnification (assumed when not explicitly specified)

#### Flow Direction
Indicates the direction of fluid flow:
- `L2R`: Left to Right
- `R2L`: Right to Left
- `L2RA`: Left to Right Assumed (for static conditions or when not specified)

#### Imaging Type
Microscopy technique used:
- `Topo`: Topography imaging
- `Trans`: Transmission imaging
- `Flat`: Flat-field imaging
- `TopoA`: Topography imaging (assumed when not specified)

#### Sequence
Unique identifier for each image in a series:
- Format: `seqXXX` where XXX is a 3-digit number
- Example: `seq001`, `seq002`, etc.

### Original to New Naming Examples
Below are examples showing how original filenames were standardized:

1. Static Condition:
   ```
   Original: H_P3-2-Static_A_70c-30T_20-21.12.21_40x-003.nd2
   New:      0Pa_A1_20dec21_40x_L2RA_FlatA_seq003.nd2
   ```

2. Flow Condition:
   ```
   Original: flow_3_1.4_L2R_20x_trans_001.nd2
   New:      1.4Pa_U_05mar19_20x_L2R_Trans_seq001.nd2
   ```

3. Pressure Transition:
   ```
   Original: flow5_8Pa_to_1.4Pa_R2L_flat_40x004.nd2
   New:      8Pa-1.4Pa_U_03apr19_40x_R2L_Flat_seq004.nd2
   ```

## Directory Structure
Data is organized into folders based on experimental conditions:
- `Static-A-1/`: First series of static experiments with TNF-α treatment
- `Static-A-2/`: Second series of static experiments with TNF-α treatment
- `1.4Pa-A-1/`: First series of 1.4Pa flow experiments with TNF-α treatment
- `1.4Pa-A-2/`: Second series of 1.4Pa flow experiments with TNF-α treatment
- `flow3_1.4Pa_18h/`: 18-hour flow experiments at 1.4Pa
- `flow5_8pa_5h/`: 5-hour flow experiments with pressure transition