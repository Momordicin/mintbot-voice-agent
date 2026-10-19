# FunASR (vendored)

`e2e-vad.h` is copied from
[modelscope/FunASR](https://github.com/modelscope/FunASR) at commit
`28a889cba5a00333dfb2dee115bd1cfab51f25e1`, path
`runtime/onnxruntime/src/e2e-vad.h` (last changed upstream in `702ec03`,
2023-11-07). It is the FSMN-VAD end-point state machine used by FunASR's C++
runtime and has no ONNX or other non-standard dependency.

Local change: one read-only accessor, `E2EVadModel::PendingEndFrame()`
(marked `mintbot`), which predicts where the open segment will end if the
current silence continues; the agent uses it to start recognition early. The
state machine itself is unchanged.

License: MIT, see the header of `e2e-vad.h`.

Known upstream quirk: `E2EVadModel::AllResetDetection()` declares local
`data_buf_size` / `data_buf_all_size` instead of resetting the members, so a
model object is not cleanly reusable after an `is_final` call. Our wrapper
never reuses one: it constructs a fresh `E2EVadModel` for every session.
