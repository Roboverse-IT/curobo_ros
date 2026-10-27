"""A frame waits, for a bounded time, for the data at its stamp (TF and others)."""

from rclpy.clock import Clock, ClockType


class StampWait:
    """
    Hands each frame to `process` once the data at its stamp is there, or at the deadline.

    A frame can arrive before the TF (or other data) at its stamp. It then waits
    here (at most one, the newest) instead of being dropped, for at most
    `max_wait_s` on the steady clock. `process(item, stamp)` is called in every
    case, also at the deadline: it does the final lookups and drops the frame
    when data is still missing.

    The guard condition and the deadline timer share `callback_group`, so the
    waiting frame is only ever changed by one callback at a time; the TF side
    and notify() (possibly other threads) only read it and trigger the guard
    condition. A TF that arrives while a watch is being registered is seen at
    the next TF message or the deadline. At most one TF watch is
    registered with the buffer: tf2 only unregisters a cancelled one on the next
    TF message, so one per frame would pile up while TF is down.
    """

    def __init__(self, node, tf_buffer, max_wait_s, process, missing_tf,
                 ready=lambda item, stamp: True, callback_group=None):
        """
        Set up the wait.

        node owns the guard condition and the deadline timer; tf_buffer is the
        tf2 buffer the frame's transforms come from. max_wait_s is the longest
        wait (s, steady clock); 0 processes every frame at once.
        process(item, stamp) is called for every submitted frame.
        missing_tf(item, stamp) returns (target, source) of a transform not yet
        available at `stamp`, or None when all are. ready(item, stamp) returns
        True when the non-TF data at `stamp` is there; call notify() when that
        data arrives. callback_group is the one of the frames' subscription.
        """
        self._tf_buffer = tf_buffer
        self._max_wait_s = max_wait_s
        self._process = process
        self._missing_tf = missing_tf
        self._ready = ready
        self._waiting = None  # (item, stamp)
        self._tf_watch = None  # future of the buffer's wait_for_transform_async
        if max_wait_s > 0.0:
            self._arrived = node.create_guard_condition(
                self._process_waiting, callback_group=callback_group)
            self._deadline = node.create_timer(
                max_wait_s, self._expire_waiting, callback_group=callback_group,
                clock=Clock(clock_type=ClockType.STEADY_TIME), autostart=False)

    def submit(self, item, stamp):
        """Process `item` now if the data at `stamp` is there, else let it wait."""
        # An older frame still waiting goes first if its data has arrived
        # meanwhile, so frames are processed in acquisition order; otherwise
        # process() drops it.
        self._process_waiting(superseded=True)

        if self._max_wait_s == 0.0 or self._available(item, stamp):
            self._process(item, stamp)
            return
        self._waiting = (item, stamp)
        self._deadline.reset()
        self._watch(item, stamp)

    def notify(self):
        """Re-check the waiting frame, to call when non-TF data has arrived (any thread)."""
        waiting = self._waiting  # the frames' callback group may clear it meanwhile
        if waiting is not None and self._ready(*waiting):
            self._arrived.trigger()

    def _available(self, item, stamp):
        return self._missing_tf(item, stamp) is None and self._ready(item, stamp)

    def _watch(self, item, stamp):
        """Trigger _process_waiting once TF covers `stamp`, or an older stamp still watched."""
        missing = self._missing_tf(item, stamp)
        if missing is not None and (self._tf_watch is None or self._tf_watch.done()):
            self._tf_watch = self._tf_buffer.wait_for_transform_async(*missing, stamp)
            self._tf_watch.add_done_callback(lambda _: self._arrived.trigger())

    def _process_waiting(self, superseded=False):
        """Process the waiting frame once its data is there (or at once if superseded)."""
        if self._waiting is None:
            return
        item, stamp = self._waiting
        if not superseded and not self._available(item, stamp):
            self._watch(item, stamp)  # what arrived was for an older frame, or not enough
            return
        self._waiting = None
        self._deadline.cancel()
        self._process(item, stamp)

    def _expire_waiting(self):
        """Process the waiting frame at the deadline, its data at the stamp or not."""
        self._deadline.cancel()
        if self._waiting is not None:
            item, stamp = self._waiting
            self._waiting = None
            self._process(item, stamp)
