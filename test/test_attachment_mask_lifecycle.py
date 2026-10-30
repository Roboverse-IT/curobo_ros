"""Mapper-aware attachment coordination, without a robot or ROS services."""

from types import SimpleNamespace

import numpy as np
import pytest

from curobo_ros.core.attachment_services import AttachmentServices
from curobo_ros.perception import ObjectDepthMasker, ObjectGeometry


class Manager:
    mapper = None

    def __init__(self):
        self.invalidations = []
        self.disabled = []

    def invalidate_tsdf_region(self, lower, upper):
        self.invalidations.append((np.asarray(lower), np.asarray(upper)))
        return 0

    def refresh_esdf(self):
        return False

    def get_scene(self):
        return object()

    def set_obstacles_enabled(self, names, enabled):
        self.disabled.append((list(names), enabled))


class Node:
    def __init__(self):
        self.object_depth_masker = ObjectDepthMasker(dilation_px=0)
        self.poses = {10: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}
        self.world_updates = []

    def get_attachment_link_pose_at_stamp(self, link, stamp):
        assert link == 'attached_object'
        return self.poses.get(stamp)

    def update_all_solvers_world(self, scene):
        self.world_updates.append(scene)


def service():
    value = object.__new__(AttachmentServices)
    value.node = Node()
    value.config_wrapper = SimpleNamespace(obstacle_manager=Manager())
    value._mapper_attachments = {}
    value._attached = {}
    return value


def test_mapper_registration_tracks_the_actual_attachment_link_at_stamp():
    attached = service()
    geometry = ObjectGeometry.cuboid([0.1, 0.1, 0.1])
    initial = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    # object_in_link is identity: at image stamp 10 it must follow link x=1.
    attached._register_mapper_attachment(
        'attached_object', 'cube', (geometry, initial, np.eye(4)))

    entry = attached.node.object_depth_masker._objects['cube']
    assert entry.pose is None
    assert entry.pose_at_stamp(10) == pytest.approx([1, 0, 0, 1, 0, 0, 0])
    assert entry.pose_at_stamp(11) is None  # never fall back to latest feedback
    assert len(attached.config_wrapper.obstacle_manager.invalidations) == 1


def test_unregister_removes_mask_and_invalidates_last_known_volume():
    attached = service()
    geometry = ObjectGeometry.cuboid([0.1, 0.1, 0.1])
    attached._register_mapper_attachment(
        'attached_object', 'cube', (geometry, [0, 0, 0, 1, 0, 0, 0], np.eye(4)))
    # Exercise the stamp resolver first, so the record retains its last
    # timestamp-valid pose for release-area invalidation.
    attached.node.object_depth_masker._objects['cube'].pose_at_stamp(10)
    assert attached._unregister_mapper_attachment('attached_object')
    assert 'cube' not in attached.node.object_depth_masker._objects
    lower, upper = attached.config_wrapper.obstacle_manager.invalidations[-1]
    np.testing.assert_allclose((lower + upper) / 2, [1.0, 0.0, 0.0])
    assert not attached._unregister_mapper_attachment('attached_object')


def test_mask_geometry_rejects_unsupported_shape_explicitly():
    attached = service()
    with pytest.raises(ValueError, match='cuboid and mesh'):
        AttachmentServices._mask_geometry(SimpleNamespace(name='sphere'))
