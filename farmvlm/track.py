"""Optional live dashboard (Trackio) for training runs; off unless TRACKIO_PROJECT is set.

Each run logs its settings, every train.jsonl and metrics.jsonl record, and Trackio's own GPU and CPU monitors; alerts
go to the same dashboard. Only numbers leave the machine, never images or weights. TRACKIO_SPACE names a private Space
to sync to; without it the dashboard stays on the training machine (`trackio show`). A dashboard failure never stops
training: the run continues with logging off.
"""
import os


class Tracker:
    def __init__(self, trackio):
        self.trackio = trackio

    def log(self, rec: dict):
        try:
            self.trackio.log({k: v for k, v in rec.items() if isinstance(v, (int, float)) and k != "step"}, step=rec.get("step"))
        except Exception as e:  # a dashboard hiccup must not end a training run
            print(f"trackio log failed: {type(e).__name__}: {e}", flush=True)

    def alert(self, title: str, text: str):
        print(f"ALERT {title}: {text}", flush=True)
        self.trackio.alert(title, text)


def start(name: str, config: dict, group: str | None = None) -> Tracker | None:
    if not os.environ.get("TRACKIO_PROJECT"):
        return None
    space = os.environ.get("TRACKIO_SPACE")
    try:
        import trackio
        trackio.init(project=os.environ["TRACKIO_PROJECT"], name=name, group=group, config=config, resume="allow",
                     space_id=space, private=True if space else None, auto_log_gpu=True, auto_log_cpu=True)
        return Tracker(trackio)
    except Exception as e:  # the dashboard is optional; training is not
        print(f"trackio off: {type(e).__name__}: {e}", flush=True)
        return None
