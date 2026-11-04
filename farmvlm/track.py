"""Optional live dashboard of training curves (Trackio) for the hyperparameter study; off unless TRACKIO_PROJECT is set.

Only each validation record goes out (pairs, loss, validation score, LR) with the run's settings: never images or weights.
TRACKIO_SPACE names a private Space to sync to; without it the dashboard stays on the training machine (`trackio show`).
"""
import os


def start(name: str, config: dict):
    """The logging callback for one run, or None when tracking is off."""
    if not os.environ.get("TRACKIO_PROJECT"):
        return None
    import trackio
    trackio.init(project=os.environ["TRACKIO_PROJECT"], name=name, config=config, resume="allow",
                 space_id=os.environ.get("TRACKIO_SPACE"))
    return trackio.log
