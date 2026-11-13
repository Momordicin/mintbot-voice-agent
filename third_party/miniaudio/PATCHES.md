# Local changes to miniaudio.h

Base: miniaudio 0.11.25. Every change is marked `MINTBOT PATCH` in `miniaudio.h`. When updating miniaudio,
re-apply them, or drop them if upstream gains an equivalent option.

## WASAPI raw capture (`ma_device_config.wasapi.rawCapture`)

Lets a capture stream open in the Windows raw processing mode (`AUDCLNT_STREAMOPTIONS_RAW`), which bypasses the
endpoint's audio enhancements (system effects), but not its always-on driver or hardware processing.

- Config: `wasapi.rawCapture` (default off). State: `ma_device.wasapi.rawCaptureActive`, set after each open,
  including reroutes, when the property was applied.
- `ma_device_init_internal__wasapi`, capture only, desktop only: reads the endpoint's
  `PKEY_Devices_AudioDevice_RawProcessingSupported` and, only when it is `VARIANT_TRUE`, calls
  `IAudioClient2::SetClientProperties` (category `Other`, options `RAW`) before `Initialize`. Requesting raw mode on an
  endpoint without it can make `Initialize` fail, hence the check (as Chromium's `audio_low_latency_input_win.cc`).
  A failed `SetClientProperties` leaves the stream in the default mode.
- The flag is carried through `ma_device_init__wasapi` and `ma_device_reinit__wasapi`, so a rerouted default device is
  checked again.
- The exclusive-mode buffer-alignment retry re-activates the client and loses the property; the agent uses shared mode.

The agent (`src/agent/audio_source.cpp`) turns it on unless `--no-raw-capture` is given, and reopens without it if
`ma_device_init` fails with it.
