# Meeting Documentation Structure

## Naming Convention
All meeting documents follow the format: `YYYY-MM-DD_X_description.format`

Where:
- `YYYY-MM-DD`: Date of the meeting
- `X`: Document type identifier (one letter)
- `description`: Brief description of the content
- `format`: File extension (e.g., md, pdf)

## Document Types
### S - Summary
- Contains meeting minutes and key discussion points
- Written during or after the meeting
- Example: `2024-11-29_S_kick_off.md`

### P - Preparation
- Meeting preparation materials
- Agenda, discussion points, preliminary work
- Example: `2024-11-29_P_experiment_design.pdf`

### O - Objectives
- Meeting goals and expected outcomes
- Action items and follow-up tasks
- Example: `2024-11-29_O_model_validation.md`

## Example Structure
```
meetings/
├── supervisors/
│   ├── Anna/
│   │   ├── 2024-11-29_P_kick_off.pdf
│   │   ├── 2024-11-29_O_kick_off.md
│   │   └── 2024-11-29_S_kick_off.md
│   └── Costanza/
│       ├── 2024-12-01_P_experiment_design.pdf
│       ├── 2024-12-01_O_experiment_goals.md
│       └── 2024-12-01_S_experiment_notes.md
```

## Best Practices
1. Always include all three document types (S, P, O) for each meeting
2. Use lowercase for descriptions
3. Use underscores to separate words
4. Keep descriptions brief but meaningful
5. Date format must always be YYYY-MM-DD