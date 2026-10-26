"""File paths in service requests may be package:// URIs into an installed package."""

import os

from ament_index_python.packages import get_package_share_directory
from curobo_msgs.srv import AddObject
from curobo_ros.core.obstacle_manager import ObstacleManager
from curobo_ros.robot.robot_description import resolve_package_uri
import pytest
import rclpy

DUCK = 'package://curobo_ros/config/config_test/Rubber_Duck.stl'


@pytest.fixture
def node():
    rclpy.init()
    node = rclpy.create_node('package_uri_test')
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_package_uri_resolves_to_the_installed_file():
    share = get_package_share_directory('curobo_ros')
    path = resolve_package_uri(DUCK)

    assert path == os.path.join(share, 'config/config_test/Rubber_Duck.stl')
    assert os.path.isfile(path)


def test_other_paths_are_unchanged():
    assert resolve_package_uri('/abs/duck.stl') == '/abs/duck.stl'
    assert resolve_package_uri('rel/duck.stl') == 'rel/duck.stl'
    assert resolve_package_uri('') == ''


@pytest.mark.parametrize('uri', ['package://no_such_package/duck.stl', 'package://'])
def test_unresolvable_package_is_a_value_error(uri):
    with pytest.raises(ValueError, match='Cannot resolve'):
        resolve_package_uri(uri)


def mesh_request(path):
    request = AddObject.Request(type=AddObject.Request.MESH, name='duck', mesh_file_path=path)
    request.pose.orientation.w = 1.0
    request.dimensions.x = request.dimensions.y = request.dimensions.z = 0.01
    return request


def test_add_object_loads_a_mesh_from_a_package_uri(node):
    manager = ObstacleManager(node, None)
    response = manager.add_object(node, mesh_request(DUCK), AddObject.Response())

    assert response.success, response.message
    assert manager.scene.mesh[0].file_path == resolve_package_uri(DUCK)


def test_add_object_reports_an_unknown_package(node):
    manager = ObstacleManager(node, None)
    response = manager.add_object(
        node, mesh_request('package://no_such_package/duck.stl'), AddObject.Response())

    assert not response.success
    assert 'no_such_package' in response.message
    assert manager.scene.mesh == []
