"""Fallback for nnsyn's optional AIM import during trainer discovery.

The course baseline does not use an AIM-tracking trainer. nnsyn's recursive
trainer discovery still imports the tracking module, however, so the name must
be importable. Install the real ``aim`` package if tracking is required.
"""


class Run:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "AIM tracking is not enabled in the course baseline. Install the "
            "real 'aim' package and use a tracking trainer if tracking is required."
        )
