"""Client-initiated NovaSynth calls — your outbound bot calls a Noveum number.

Fill in sections 1 and 2. Section 3 is the loop and needs no changes.

How it works: each run id you were given is a (persona x scenario) test case.
This script claims a run, waits for the platform to park a synthetic persona
on one of your Noveum numbers, then hands that number to YOUR dialler. Your
bot calls it, talks to the persona, and the platform records and scores the
conversation. Repeat for every run id, CONCURRENCY at a time.

Requirements:
    pip install noveum-trace
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from noveum_trace.novasynth import Call, CallQueue

# --- 1. Your details --------------------------------------------------------

NOVEUM_API_KEY = "nv_..."  # or leave None and set the NOVEUM_API_KEY env var
ORGANIZATION_SLUG = "your-org-slug"  # shown in the dashboard URL
RUN_IDS = [
    # Run ids from the inbound batch you created (dashboard -> batch -> run ids)
    "run_id_1",
    "run_id_2",
]
# How many calls to run at once. Must not exceed the number of Noveum phone
# numbers provisioned for you (extra workers just wait for a free number).
CONCURRENCY = 2


# --- 2. Your dialler --------------------------------------------------------


def place_call(call: Call) -> None:
    """Make your outbound bot dial ``call.dial_number``.

    What you get:
        call.dial_number        E.164 number to dial, e.g. "+918065481242"
        call.profile            dict of this test's customer profile (name,
                                phone, account details, situation...) — pass it
                                to your bot the same way you would for a real
                                customer
        call.persona_name       for your logs
        call.scenario_name      for your logs
        call.seconds_remaining  dial before this hits 0 (window is ~5 min)

    Rules:
      * Dial within ``call.seconds_remaining``. Late calls are rejected and
        the run is marked expired.
      * Your bot must speak first. The persona answers and then waits for
        your bot to open the conversation, like a real customer picking up.
      * Return when the call has ended, or return right away — either is
        fine, the loop waits for the platform to mark the run finished.
      * Raise on a dial failure (busy, no route...). The run is skipped and
        the number is released when its window closes.
    """
    raise NotImplementedError("dial call.dial_number with your bot here")


# --- 3. The loop (no changes needed) ---------------------------------------


def handle(call: Call) -> None:
    print(
        f"[{call.run_id}] dial {call.dial_number} "
        f"({call.persona_name} / {call.scenario_name}, "
        f"{call.seconds_remaining:.0f}s left)"
    )
    try:
        place_call(call)
    except Exception as exc:  # noqa: BLE001 — a dial failure must not stop the batch
        print(f"[{call.run_id}] dial failed: {exc}")
        return
    print(f"[{call.run_id}] finished: {call.wait_until_finished()}")


if __name__ == "__main__":
    queue = CallQueue(
        RUN_IDS,
        api_key=NOVEUM_API_KEY,
        organization_slug=ORGANIZATION_SLUG,
    )
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        in_flight = set()
        for call in queue.iter_calls():
            in_flight.add(pool.submit(handle, call))
            if len(in_flight) == CONCURRENCY:
                # Claim the next run only once a worker is free — a run claimed
                # early burns its dial window waiting for a worker.
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()  # surface errors handle() did not catch
        for future in wait(in_flight).done:
            future.result()
    print("summary:", queue.summary())  # e.g. {'completed': 5, 'expired': 1}
