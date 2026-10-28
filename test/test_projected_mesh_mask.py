"""ProjectedMeshMask rasterizes generic link-local geometry on the GPU."""

import xml.etree.ElementTree as ET

import pytest
import torch
import trimesh

from curobo_ros.perception.projected_mesh_mask import (
    ProjectedMeshMask, resolve_urdf_mesh_paths)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='needs CUDA')


def test_package_mesh_uris_resolve_from_a_project_source_tree(tmp_path):
    """Generated URDFs can reuse meshes from uninstalled source packages."""
    project = tmp_path / 'cell'
    urdf_dir = project / 'config'
    mesh = (project / 'src' / 'vendor' / 'custom_description'
            / 'meshes' / 'link.stl')
    urdf_dir.mkdir(parents=True)
    mesh.parent.mkdir(parents=True)
    mesh.write_text('solid link\nendsolid link\n')
    urdf = urdf_dir / 'robot.urdf'
    urdf.write_text(
        '<robot name="test"><link name="link"><visual><geometry>'
        '<mesh filename="package://custom_description/meshes/link.stl"/>'
        '</geometry></visual><collision><geometry>'
        '<mesh filename="package://missing_description/meshes/missing.stl"/>'
        '</geometry></collision></link></robot>')

    resolved, unresolved = resolve_urdf_mesh_paths(str(urdf))

    filenames = [node.get('filename') for node in
                 ET.parse(resolved).getroot().iter('mesh')]
    assert resolved != str(urdf)
    assert filenames[0] == str(mesh)
    assert filenames[1] == (
        'package://missing_description/meshes/missing.stl')
    assert unresolved == [
        'package://missing_description/meshes/missing.stl']


def test_mesh_uri_resolver_rejects_a_missing_urdf(tmp_path):
    """A missing source URDF fails before kinematics construction."""
    with pytest.raises(ValueError, match='URDF'):
        resolve_urdf_mesh_paths(str(tmp_path / 'missing.urdf'))


class Mesh:
    """Minimal cuRobo Mesh-compatible wrapper around a trimesh object."""

    def __init__(self, mesh):
        """Store source geometry for repeatable independent loads."""
        self._mesh = mesh

    def get_trimesh_mesh(self, process=False, transform_with_pose=True):
        """Return an independent mesh, as cuRobo's Mesh method does."""
        return self._mesh.copy()


class MissingMesh:
    """A configured URDF visual whose package asset is unavailable."""

    file_path = 'package://missing_description/meshes/link.stl'

    def get_trimesh_mesh(self, process=False, transform_with_pose=True):
        """Raise the error produced by a missing package mesh asset."""
        raise FileNotFoundError(self.file_path)


def backend(margin=0.0):
    """Build one 20 cm cube link with an optional geometric margin."""
    cube = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    return ProjectedMeshMask(
        ['cube_link'], [Mesh(cube)], 'cuda', margin=margin,
        raster_stride=1)


def render(segmenter, x=0.0, depth=None):
    """Render a cube one metre in front of an identity optical camera."""
    if depth is None:
        depth = torch.ones((64, 64), dtype=torch.float32, device='cuda')
    return segmenter.mask(
        depth,
        (60.0, 60.0, 32.0, 32.0),
        torch.eye(4, dtype=torch.float32, device='cuda'),
        torch.tensor([[x, 0.0, 1.0]], dtype=torch.float32, device='cuda'),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]],
                     dtype=torch.float32, device='cuda'),
    ).clone()


@requires_cuda
def test_projected_cube_has_a_bounded_silhouette():
    """Rasterize the link at the pinhole projection of its base-frame pose."""
    mask = render(backend())

    assert mask[32, 32]
    assert not mask[0, 0]
    assert 100 < mask.sum().item() < 400


@requires_cuda
def test_mask_follows_the_link_pose_and_valid_depth():
    """The same geometry moves with FK and never masks invalid source depth."""
    segmenter = backend()
    centered = render(segmenter)
    shifted = render(segmenter, x=0.2)
    depth = torch.ones((64, 64), dtype=torch.float32, device='cuda')
    depth[32, 32] = 0.0
    invalid = render(segmenter, depth=depth)

    centered_u = centered.nonzero()[:, 1].float().mean()
    shifted_u = shifted.nonzero()[:, 1].float().mean()
    assert shifted_u > centered_u + 8.0
    assert not invalid[32, 32]


@requires_cuda
def test_mesh_margin_expands_the_silhouette():
    """Inflating link-local hulls expands coverage without pixel dilation."""
    plain = render(backend(margin=0.0))
    inflated = render(backend(margin=0.04))

    assert inflated.sum() > plain.sum()
    assert torch.all(inflated[plain])


@requires_cuda
def test_auto_selected_missing_geometry_is_skipped_but_explicit_is_strict():
    """Auto discovery tolerates absent visuals; an explicit list does not."""
    cube = trimesh.creation.box(extents=[0.2, 0.2, 0.2])

    automatic = ProjectedMeshMask(
        ['missing', 'cube'], [MissingMesh(), Mesh(cube)], 'cuda', strict=False)

    assert automatic.link_names == ['cube']
    assert automatic.skipped_link_names == ['missing']
    with pytest.raises(ValueError, match='cannot be loaded'):
        ProjectedMeshMask(
            ['missing'], [MissingMesh()], 'cuda', strict=True)


@pytest.mark.parametrize('kwargs, match', [
    ({'margin': -0.1}, 'margin'),
    ({'raster_stride': 0}, 'stride'),
])
@requires_cuda
def test_configuration_is_validated(kwargs, match):
    """Reject invalid geometric settings before receiving a depth frame."""
    cube = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    with pytest.raises(ValueError, match=match):
        ProjectedMeshMask(['cube'], [Mesh(cube)], 'cuda', **kwargs)
