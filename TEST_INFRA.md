# E2E Test Infra: DeepFeik

## Test Philosophy
- **Opaque-box & Requirement-driven**: All E2E test cases derive strictly from `ORIGINAL_REQUEST.md` and user-facing requirements, never relying on internal implementation details.
- **Methodology**: Systematic 4-tier testing combining Category-Partition, Boundary Value Analysis (BVA), Pairwise Combinatorial Testing, and Realistic Workload Sessions.
- **Deterministic & Headless Execution**: Zero dependency on physical webcam hardware or physical display monitors. Enabled via `MockVideoCapture` / `MockCameraSource` and `QT_QPA_PLATFORM=offscreen`.

## Feature Inventory
| # | Feature | Source | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---------|--------|:------:|:------:|:------:|:------:|
| 1 | F1. Webcam Video Acquisition | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 2 | F2. Video Source Abstraction & Mock | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 3 | F3. Real-Time Face Detection & Landmarks | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 4 | F4. Source Image Loading (File Picker) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 5 | F5. Source Image Loading (Clipboard Ctrl+V) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 6 | F6. Face Validation & Error Dialogs | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 7 | F7. Delaunay Mesh Piecewise Affine Warping | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 8 | F8. Skin Tone & Color Matching | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 9 | F9. Seamless Edge Feathering & Blending | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 10 | F10. Real-Time CPU Pipeline Coordination | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 11 | F11. Live Preview GUI Display | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| 12 | F12. GUI Controls & Status Feedback | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| 13 | F13. Asynchronous Video Recording | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 14 | F14. Wall-Clock Paced Recording & Finalization | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 15 | F15. Clean Application & Device Shutdown | Acceptance Criteria | 5 | 5 | ✓ | ✓ |
| 16 | F16. Pip-Installable Packaging & Entry Point | Acceptance Criteria | 5 | 5 | ✓ | ✓ |

## Test Architecture
- **Test Runner**: `pytest` running with `pytest-qt` in headless mode (`QT_QPA_PLATFORM=offscreen`).
- **Pass/Fail Semantics**: All test cases must pass with exit code 0. Zero warnings treated as errors where feasible.
- **Mock Video Source (`MockCameraSource`)**:
  - Mode 1: Static image loop with known facial geometry.
  - Mode 2: Procedural animated face with dynamic head rotation, translation, eye-blinking, and mouth movement.
  - Mode 3: Pre-recorded MP4 video clip playback.
  - Mode 4: Scripted timeline with fault injection (e.g. camera disconnection at frame 45).
- **Directory Layout**:
  - `tests/conftest.py`: Global fixtures for QApplication, mock camera, and synthesized face images.
  - `tests/e2e/test_tier1_features.py`: 24 tests covering happy-path isolated features.
  - `tests/e2e/test_tier2_boundaries.py`: 24 tests covering corner cases and boundary conditions.
  - `tests/e2e/test_tier3_pairwise.py`: 18 tests covering feature interactions.
  - `tests/e2e/test_tier4_scenarios.py`: 6 multi-step end-to-end user workflows.
  - `tests/benchmarks/test_benchmark_cpu.py`: CPU FPS and latency percentiles verification.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Standard Video Call Session | F1, F3, F4, F7, F8, F9, F10, F11, F12, F15 | High |
| 2 | Presenter with Dynamic Source Swapping | F4, F5, F6, F7, F10, F11, F12 | High |
| 3 | Recording Full Swap Demonstration to MP4 | F1, F4, F7, F9, F11, F12, F13, F14, F15 | High |
| 4 | Interrupted Stream & Recovery | F1, F2, F3, F10, F11, F15 | High |
| 5 | Sustained Load & Resource Leak Check | F1, F7, F9, F10, F11, F13, F14 | Extreme |
| 6 | Headless Automated CI Smoke Pipeline | F2, F4, F5, F7, F10, F13, F16 | Medium |

## Coverage Thresholds
- **Tier 1 (Feature Coverage)**: 24 test cases (>= 5 per major requirement area).
- **Tier 2 (Boundary & Corner Cases)**: 24 test cases (extreme inputs, zero faces, transparent alpha, disconnects).
- **Tier 3 (Cross-Feature Combinations)**: 18 test cases (pairwise interactions).
- **Tier 4 (Real-World Workloads)**: 6 multi-step application scenarios.
- **Performance Threshold**: Must maintain >= 15.0 FPS on CPU throughout active face swapping and recording.
