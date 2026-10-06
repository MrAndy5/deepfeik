# Original User Request

## 2026-09-28T17:16:16Z

A Python desktop application that performs real-time face-swapping using the computer's live webcam feed. The user loads a source photo (via file picker or clipboard paste), and the app detects the face in that photo and seamlessly composites it onto their own face in the live camera stream in real time, preserving natural facial structure, alignment, and blending. The app must also allow recording a video clip of the live swapped session.

The app should be pip-installable and cross-platform friendly (Windows first, but designed with future Android APK portability in mind — avoid hard platform dependencies where possible).

Working directory: ~/teamwork_projects/deepfake_face_swap
Integrity mode: benchmark (no restrictions — any library, model, or approach that works)

## Requirements

### R1. Live camera face swap
The app opens the computer's default webcam and continuously detects the user's face in the live feed. When a source face image is loaded, the app replaces the user's face in real time with the source face, aligning facial landmarks (eyes, nose, mouth, jaw) and blending the result so it looks natural within the live frame. The frame rate should remain interactive (target ≥ 15 FPS on a modern laptop).

### R2. Source image loading
The user can load a source face photo either via a file picker dialog (JPG/PNG) or by pasting an image directly from the clipboard (Ctrl+V). If no face is detected in the loaded image, the app should display a clear error message to the user.

### R3. Video recording
The user can start and stop recording a video clip of the live face-swapped output. The recording is saved as a standard video file (e.g., MP4 or AVI) to a user-chosen location.

### R4. GUI
The app must have a graphical user interface (not CLI-only) that shows: the live swapped camera feed, controls to load a source image (file picker button + paste shortcut), and start/stop recording controls.

## Acceptance Criteria

### Live face swap quality
- [ ] The app launches and shows the live webcam feed within 5 seconds of starting
- [ ] When a source image with a visible face is loaded, the face swap activates within 2 seconds
- [ ] The swapped face is geometrically aligned to the user's facial landmarks (no obvious large misalignment of eyes/nose/mouth)
- [ ] The blending boundary between swapped face and surrounding skin is not a hard cut (some feathering or color correction is applied)
- [ ] Performance: live preview updates at ≥ 15 FPS during active face swap on a modern laptop CPU (GPU optional bonus)

### Source image loading
- [ ] File picker opens and successfully loads a JPG or PNG with a face
- [ ] Pasting an image from clipboard (Ctrl+V) loads the image and activates the swap
- [ ] If the loaded image has no detectable face, the app shows an error message (no crash)

### Video recording
- [ ] Clicking start recording begins saving the swapped video frames to disk
- [ ] Clicking stop recording finalizes and closes the video file, which is playable in a standard media player
- [ ] The recorded video visually matches the live preview output

### GUI and usability
- [ ] All core controls (load image, paste, start/stop record) are accessible from the main window
- [ ] The app exits cleanly without hanging or crashing the camera

## 2026-10-02T06:25:51Z

This is a continuation improvement pass on the deepfeik real-time face-swap desktop app at `C:\Users\andre\Desktop\deepfeik`. The app is already working. Two tasks remain:

**1. Skin tone adaptation** — The existing Reinhard LAB color transfer in `src/deepfeik/core/color.py` corrects skin-tone mismatch between the source face photo and the live camera frame. This needs to be strengthened so that per-frame updates continuously adapt the swapped face's color to match the user's live skin as lighting conditions change. The source face should look like it truly belongs to the user's face regardless of skin tone difference.

**2. GitHub release + APK build** — Push the code to GitHub under user `Mr_andy5`, trigger the APK CI build, and ensure the release is visible at `https://github.com/Mr_andy5/deepfeik/releases`.

Working directory: C:\Users\andre\Desktop\deepfeik
Integrity mode: benchmark (no restrictions)

## Existing Codebase

```
src/deepfeik/
├── core/
│   ├── color.py        ← ReinhardColorMatcher (Reinhard LAB transfer, cached source stats)
│   ├── engine.py       ← FaceSwapEngine.process_frame — calls color_matcher.match() per frame
│   ├── blender.py      ← FeatheredAlphaBlender
│   ├── warper.py       ← PiecewiseAffineWarper
│   ├── face_mesh.py    ← FaceMeshTracker
│   └── landmarks.py
├── pipeline/           ← WebcamSource, LatestFrameBuffer
├── gui/
│   ├── main_window.py  ← Modern PyQt5 UI (dark glassmorphism)
│   ├── camera_thread.py
│   └── clipboard.py
└── __main__.py

android/
├── main.py             ← Kivy app for Android
└── buildozer.spec

.github/workflows/
└── build_apk.yml       ← GitHub Actions: builds APK on Ubuntu, uploads as release

tests/unit/             ← 179 tests, all passing
pyproject.toml
requirements.txt
.gitignore
```

Git status: 1 commit (`28fea31`), no remote set. `gh` CLI is at `C:\Program Files\GitHub CLI\gh.exe` — the user is ready to complete a browser-based login if a popup appears. Run `& "C:\Program Files\GitHub CLI\gh.exe" auth login --web --hostname github.com` to trigger it.

## Requirements

### R1. Per-frame adaptive skin tone matching
Improve `src/deepfeik/core/color.py` and/or `src/deepfeik/core/engine.py` so that skin tone transfer is re-computed or adapted every frame using the user's live face region as the target reference (not just a static cached source stat). The result: when the user's lighting changes or their skin tone differs from the source photo, the blend still looks natural. The fix must not drop performance below 15 FPS overall.

### R2. GitHub push and APK release
Authenticate `gh` CLI for `Mr_andy5` (using `& "C:\Program Files\GitHub CLI\gh.exe" auth login --web --hostname github.com` — user will handle the browser popup), create the public repo `Mr_andy5/deepfeik` if it doesn't exist, push all committed code, tag it `v1.0.0`, and ensure the GitHub Actions workflow (`.github/workflows/build_apk.yml`) triggers. Verify the release appears at `https://github.com/Mr_andy5/deepfeik/releases`. Use the full path `C:\Program Files\GitHub CLI\gh.exe` for all gh commands.

## Acceptance Criteria

### Skin tone
- [ ] In `engine.py`'s `process_frame`, the target face region LAB stats are computed from the live frame (not only at source load time) to drive the color transfer
- [ ] All 179 existing unit tests still pass after the change
- [ ] A quick timing check shows `process_frame` still executes in under 20ms per frame on the local machine

### GitHub
- [ ] `C:\Program Files\GitHub CLI\gh.exe` is authenticated to `Mr_andy5`
- [ ] `https://github.com/Mr_andy5/deepfeik` exists with source code
- [ ] Tag `v1.0.0` is pushed, triggering the Actions workflow
- [ ] A release exists (or is in progress) at `https://github.com/Mr_andy5/deepfeik/releases`
