from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from curobo_ros.robot.joint_control_strategy import JointCommandStrategy, RobotState
from curobo_ros.robot.joint_feedback import JointFeedback


class JointPoseStrategy(JointCommandStrategy):
    '''Joint-POSE control: command target POSITIONS (a JointTrajectory whose points
    carry positions only; the driver interpolates to each pose). Same message type
    and topics as joint_speed, but no velocity/acceleration setpoints — for drivers
    that follow position references rather than streamed speeds.

    Descriptor strategy_params: command_topic, state_topic (opt), joint_states_topic (opt).
    '''

    def __init__(self, node, dt, description=None):
        super().__init__(node, dt, description)
        # self.dt (base class) is already the resolved interpolation_dt —
        # curobo_ros is the single authority on trajectory pacing (see
        # resolve_interpolation_dt).

        command_topic = self.params.get('command_topic', '/execute_trajectory')
        self.pub_trajectory = node.create_publisher(JointTrajectory, command_topic, 10)

        state_topic = self.params.get('state_topic')
        if state_topic:
            self.sub_trajectory_state = node.create_subscription(
                Float32, state_topic, self.callback_trajectory_state, 10,
                callback_group=MutuallyExclusiveCallbackGroup())

        self.feedback = JointFeedback(description, node.get_logger())
        joint_states_topic = self.params.get('joint_states_topic')
        if joint_states_topic:
            self.sub_joint_state = node.create_subscription(
                JointState, joint_states_topic, self.callback_joint_pose, 10,
                callback_group=MutuallyExclusiveCallbackGroup())

    def send_trajectrory(self):
        with self.buffer_lock:
            self.robot_state = RobotState.RUNNING
            msg = JointTrajectory()
            msg.joint_names = self.description.joint_state_map.driver_names(self.joint_names) \
                if self.description is not None else list(self.joint_names)

            if len(self.position_command) == 0:
                self.trajectory_progression = 1.0

            for i in range(len(self.position_command)):
                point = JointTrajectoryPoint()
                point.positions = self.position_command[i]   # positions only
                point.time_from_start = Duration(
                    sec=int(self.dt * i), nanosec=int((self.dt * i % 1) * 1e9))
                msg.points.append(point)

            self.position_command = []
            self.vel_command = []
            self.accel_command = []
            self.trajectory_progression = 0.0

        self.pub_trajectory.publish(msg)

    def get_joint_pose(self):
        with self.buffer_lock:
            return self.feedback.read()[0]

    def get_joint_name(self):
        """Return the model's joint names, the order of get_joint_pose()."""
        with self.buffer_lock:
            return self.feedback.names

    def get_joint_state_age(self):
        with self.buffer_lock:
            return self.feedback.age()

    def stop_robot(self):
        with self.buffer_lock:
            self.vel_command = []
            self.position_command = []
            self.accel_command = []
            self.command_index = 0
            self.trajectory_progression = 0.0
            self.robot_state = RobotState.STOPPED
        self.pub_trajectory.publish(JointTrajectory())

    def callback_trajectory_state(self, msg):
        with self.buffer_lock:
            self.trajectory_progression = msg.data

    def callback_joint_pose(self, msg):
        with self.buffer_lock:
            self.feedback.update(msg)

    def get_progression(self):
        with self.buffer_lock:
            return self.trajectory_progression
