"""A node of its own, spun on a thread of its own, for high-rate subscriptions."""

import threading

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor


class ListenerThread:
    """
    A node spun on its own thread, so its callbacks never wait for the owner's.

    High-rate topics (on a UR, TF and /joint_states arrive at about 500 Hz
    each) handled on the owner's executor queue up behind its slow callbacks
    (e.g. segmenting a depth frame), and arrive there late. Subscriptions
    created on `node` are handled here as they arrive, on this thread.

    Create the subscriptions only once what their callbacks use exists: the
    thread spins from the start. Call destroy() before the owner's
    destroy_node().
    """

    def __init__(self, name):
        """
        Create the node `name` and start spinning it.

        Global arguments (namespace, remaps) apply to it as to the owner; the
        name in cli_args overrides a __node remap meant for the owner (as tf2's
        own listener node does).
        """
        self.node = rclpy.create_node(
            name, cli_args=['--ros-args', '-r', f'__node:={name}'],
            start_parameter_services=False)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._spin, name=name, daemon=True)
        self._thread.start()

    def _spin(self):
        try:
            self._executor.spin()
        except ExternalShutdownException:
            pass  # the context was shut down (e.g. Ctrl-C) before destroy()

    def destroy(self):
        """Stop the thread, then destroy the node."""
        self._executor.shutdown()
        self._thread.join()
        self.node.destroy_node()
