# Unified Recording Settings UI

## Direction

The settings view uses a restrained flat workstation aesthetic: white canvas,
soft gray grouping surfaces, no decorative borders, and fixed-height controls.
Device selection and speech detection belong to one `Recording and detection`
section because both define the microphone input before a session starts.
Model management remains a separate offline-inference section.

## Interaction

The microphone selector, active ASR selector, and advanced model-kind selector
use one custom listbox component. Its trigger is 42 px tall and aligns exactly
with adjacent actions. It supports pointer input, Arrow keys, Home/End,
Enter/Space, Escape, disabled options, click-outside dismissal, and visible
selected state. Refresh remains an icon-only command with a tooltip and a
spinning busy state.

Speech detection is embedded below the device row rather than opening a second
side panel. Automatic/manual mode, threshold, reset, and live RMS/peak values
retain their current behavior and lock together while recording. Wide layouts
place mode and threshold controls side by side; narrow layouts stack them.

## Acceptance

- Settings has one recording section and one model section.
- No standalone speech-detection button or drawer remains.
- All settings-page selects use the custom listbox.
- Device selector and refresh button share height and vertical alignment.
- Existing loading, error, recording-lock, save, and selection events remain
  covered by component tests.
