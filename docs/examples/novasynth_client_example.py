"""Client-initiated NovaSynth calls — your dialler calls us.

In the normal flow NovaSynth dials your voice agent. Here it is reversed: your
own dialler places the PSTN call and a Noveum-hosted synthetic persona answers.
A session has to be parked in the LiveKit room before the call arrives (~35s),
and one number maps to one room, so the platform decides when each run is
dialable and this loop waits to be told.

Illustrative only until the platform ships /v1/novasynth/*. The behaviour it
depends on is covered in tests/unit/novasynth/.
"""

from concurrent.futures import ThreadPoolExecutor

import noveum_trace
from noveum_trace.novasynth import Call, CallQueue

noveum_trace.init(project="voice-qa", api_key="your-noveum-api-key")

# Run ids come back from whatever created the batch (NovaEval CLI, dashboard,
# or API). Keep them: they are how you reconcile the batch afterwards. The two
# blocks below are separate batches — a batch that has already run is terminal,
# so re-polling its ids would yield nothing.
SEQUENTIAL_RUN_IDS = ["run_a", "run_b", "run_c"]
CONCURRENT_RUN_IDS = ["run_d", "run_e", "run_f"]


class DialFailed(Exception):
    """Whatever your telephony provider raises."""


def place(to: str, variables: dict) -> str:
    """Stand-in for your dialler. Returns the provider's call id."""
    raise DialFailed("no trunk configured in this example")


# --- Sequential: one call at a time ----------------------------------------

with CallQueue(SEQUENTIAL_RUN_IDS, batch_run_id="br_01JABCXYZ") as q:
    for call in q.iter_calls():
        # call.agent_variables is the agent-facing half of the profile only.
        # The persona's situational context never leaves the platform — if the
        # agent under test knew it, it would score itself far too well.
        print(f"dial {call.dial_number} within {call.seconds_remaining:.0f}s")
        try:
            sid = place(to=call.dial_number, variables=call.agent_variables)
        except DialFailed as exc:
            # Frees the number now instead of at the end of the dial window.
            call.report_failed(reason=str(exc), code="busy")
            continue
        print(call.run_id, "->", call.wait_until_finished(provider_call_id=sid))

    print(q.summary())  # every run id in the batch, counted by last status


# --- Concurrent: feed the same generator to a pool -------------------------
# Size the pool to your provisioned numbers (realistically 2-3) — the queue
# yields runs only as the platform frees a number, so a larger pool just idles.


def handle(call: Call) -> None:
    try:
        sid = place(to=call.dial_number, variables=call.agent_variables)
    except DialFailed as exc:
        call.report_failed(reason=str(exc), code="busy")
        return
    call.wait_until_finished(provider_call_id=sid)


with CallQueue(CONCURRENT_RUN_IDS) as q, ThreadPoolExecutor(3) as pool:
    list(pool.map(handle, q.iter_calls()))
    print(q.summary())

# Releases the SDK client and its background batch processor.
noveum_trace.shutdown()
