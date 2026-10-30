# Retained timing deviation and resume

The first fixed-seed batch was registered with a 900-second process/batch wall
budget. Its first experiment, `attention_regularized_s1`, actually completed in
1124.3837074 seconds (1098.3893508 summed fitting wall seconds). The subprocess
returned success; the parent then detected that no batch budget remained and
stopped before starting the next experiment. Therefore the 15-minute cap was
exceeded: it was not a strict end-to-end enforced cap on this observed run.

All fold predictions, epoch histories, actual timing and the parent timeout log
are retained. The cause of the long wall-clock interval is not established; no
unmeasured CPU-only time is substituted. Scientific results are not discarded
because of runtime or seed quality.

At 2026-10-29T09:19Z the remaining three already-registered seed configurations
were resumed in `stability-configs-resume1.json`, with a new 900-second batch
bound, while remaining within the original practical 60-minute additional
research window (approximately 08:45--09:45Z) and the absolute six-hour cap.
The completed seed was excluded from that list and not retrained. The inference
resource measurement is a separate fresh-process validation run; these training
times are not evidence about reference-machine inference eligibility.
