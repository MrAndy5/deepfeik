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
