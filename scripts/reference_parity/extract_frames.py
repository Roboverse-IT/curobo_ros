"""Extract exact recorded depth, camera TF, and interpolated joint states."""
import sys
import numpy as np
import rosbag2_py
import tf2_ros
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_msgs.msg import TFMessage

BAG = '/opt/bags/live_moving_redcube_20261026'
DEPTH = '/wrist_camera/aligned_depth_to_color/image_raw'
INFO = '/wrist_camera/aligned_depth_to_color/camera_info'
NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'finger_joint']
TARGETS = tuple(np.arange(0.5, 89.5, 1.0))


def reader(topics):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id='mcap'),
           rosbag2_py.ConverterOptions('cdr', 'cdr'))
    r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    while r.has_next():
        yield r.read_next()


buf = tf2_ros.Buffer(cache_time=Duration(seconds=120))
joint_t, joint_q, K = [], [], None
for topic, payload, _ in reader(['/tf', '/tf_static', '/joint_states', INFO]):
    if topic in ('/tf', '/tf_static'):
        for tr in deserialize_message(payload, TFMessage).transforms:
            (buf.set_transform_static if topic == '/tf_static' else buf.set_transform)(tr, 'bag')
    elif topic == '/joint_states':
        msg = deserialize_message(payload, JointState)
        idx = {name: i for i, name in enumerate(msg.name)}
        if all(name in idx for name in NAMES):
            joint_t.append(Time.from_msg(msg.header.stamp).nanoseconds)
            joint_q.append([msg.position[idx[name]] for name in NAMES])
    elif K is None:
        K = np.array(deserialize_message(payload, CameraInfo).k, dtype=np.float64).reshape(3, 3)
joint_t, joint_q = np.asarray(joint_t), np.asarray(joint_q)
order = np.argsort(joint_t)
joint_t, joint_q = joint_t[order], joint_q[order]
print('joint_samples', len(joint_t), 'K', K.reshape(-1).tolist(), flush=True)

picked = {}
first = None
for _, payload, _ in reader([DEPTH]):
    msg = deserialize_message(payload, Image)
    ns = Time.from_msg(msg.header.stamp).nanoseconds
    if first is None:
        first = ns
    seconds = (ns - first) * 1e-9
    for target in TARGETS:
        delta = abs(seconds - target)
        if target not in picked or delta < picked[target][0]:
            picked[target] = (delta, msg)

out = {'K': K, 'joint_names': np.array(NAMES)}
for i, target in enumerate(TARGETS):
    delta, msg = picked[target]
    ns = Time.from_msg(msg.header.stamp).nanoseconds
    tr = buf.lookup_transform('base_link', msg.header.frame_id, Time(nanoseconds=ns))
    q = np.array([np.interp(ns, joint_t, joint_q[:, j]) for j in range(len(NAMES))], dtype=np.float32)
    depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
    out[f'depth_{i}'] = depth
    out[f'stamp_ns_{i}'] = np.int64(ns)
    out[f'frame_id_{i}'] = np.array(msg.header.frame_id)
    out[f'joint_q_{i}'] = q
    out[f'camera_xyz_{i}'] = np.array([tr.transform.translation.x, tr.transform.translation.y,
                                      tr.transform.translation.z], dtype=np.float32)
    out[f'camera_xyzw_{i}'] = np.array([tr.transform.rotation.x, tr.transform.rotation.y,
                                       tr.transform.rotation.z, tr.transform.rotation.w], dtype=np.float32)
    print('frame', i, 'target_s', target, 'offset_s', delta, 'stamp_ns', ns,
          'valid', int((depth > 0).sum()), 'q', q.tolist(), flush=True)
out['n'] = len(TARGETS)
np.savez_compressed(sys.argv[1], **out)
