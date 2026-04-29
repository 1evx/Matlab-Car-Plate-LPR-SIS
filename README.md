# Car Plate Recognise System

MATLAB starter project for a License Plate Recognition (LPR) and State Identification System (SIS) focused on Malaysian vehicle registrations. The scaffold is organized for group coursework: modular pipeline code, a runnable GUI, dataset conventions, tests, experiments, and submission assets.

## Project Goals

- Detect license plates from vehicle images under varied lighting, distance, and background conditions.
- Segment and recognize plate characters using classical computer-vision techniques.
- Infer the registered Malaysian state or special category from the recognized plate text.
- Provide a GUI for demonstrations and coursework presentation.

## Toolboxes

Recommended MATLAB release: `R2023a` or newer

Required:
- Image Processing Toolbox

Optional:
- Statistics and Machine Learning Toolbox

If the statistics toolbox is unavailable, the character recognizer falls back to prototype-based nearest-neighbor matching over handcrafted character features. The scaffold does not require Computer Vision Toolbox for the current baseline implementation.

## Quick Start

1. Open the folder in MATLAB.
2. Run:

```matlab
startup
```

3. Launch the GUI:

```matlab
main
```

4. Or run the pipeline directly on one image:

```matlab
result = main("data/samples/example.jpg");
disp(result.recognizedText)
disp(result.stateInfo)
```

5. If you want the classical feature-based character model ready in advance:

```matlab
trainCharacterClassifier
```

## Notes About the GUI

This scaffold ships with a programmatic MATLAB app class in [src/gui/LPRStateApp.m](/e:/Github%20Project/Car-Plate-Recognise-System/src/gui/LPRStateApp.m). It is source-control friendly and runnable immediately.

If your group prefers a `.mlapp` file for App Designer, use this class as the functional reference while recreating the visual layout in App Designer later. The pipeline API is already separated so the GUI layer can be swapped without changing the detection logic.

## Folder Layout

```text
Car-Plate-Recognise-System/
├─ CarPlateRecogniseSystem.prj
├─ startup.m
├─ data/
├─ docs/
├─ src/
├─ tests/
├─ experiments/
├─ models/
└─ deliverables/
```

## Dataset Convention

Recommended image naming:

```text
vehicleType_plateCategory_lighting_distance_index.jpg
```

Example:

```text
car_standard_day_near_001.jpg
```

Annotation CSV columns:

- `image_name`
- `vehicle_type`
- `plate_category`
- `expected_text`
- `expected_state`
- `bbox_x`
- `bbox_y`
- `bbox_w`
- `bbox_h`
- `notes`

See [data/annotations/sample_annotations.csv](/e:/Github%20Project/Car-Plate-Recognise-System/data/annotations/sample_annotations.csv).

## Pipeline Overview

The processing flow is:

1. Normalize lighting and enhance contrast
2. Generate and rank likely license plate candidates with multi-branch morphology and connected-component analysis
3. Rectify and crop the plate
4. Segment characters from single-line or two-row layouts
5. Recognize characters with a classical feature-based classifier and Malaysian syntax constraints
6. Infer the state/series with Malaysian registration rules

Key public entry points:

- `runLPRPipeline`
- `detectPlateRegion`
- `segmentCharacters`
- `recognizeCharacters`
- `identifyState`
- `LPRStateApp`

## Running Tests

```matlab
startup
results = runtests({'tests/unit','tests/integration'});
table(results)
```

Or use the helper:

```matlab
runAllTests
```

## Team Workflow

- Keep production code in `src/`
- Keep one-off experiments in `experiments/`
- Store trained models in `models/trained/`
- Put screenshots and demo assets in `deliverables/`
- Record failed cases and analysis in `experiments/results/`

## Suggested Next Steps

- Build a real dataset across cars, buses, and motorcycles
- Expand Malaysian rules for more special series
- Tune plate candidate scoring with your own collected images
- Recreate the app layout in App Designer if your lecturer specifically wants `.mlapp`
