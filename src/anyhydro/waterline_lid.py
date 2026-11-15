# SPDX-License-Identifier: MPL-2.0
"""Bounded waterline extraction for the shared ANYmesher planar triangulator.

This is an optional mathematical BEM surface. It does not remesh the hull.
Native/geometry dependencies are imported only during explicit generation.
"""
from collections import Counter, defaultdict
import math

import numpy as np

from .wave_surface_fields import WaveFieldError


def generate_waterline_lid(vertices, faces, *, cell_size, max_panels, max_work_bytes):
    """Cover one simple, exactly z=0 waterline with constrained triangles.

    Boundary vertices/segments are passed unchanged to ANYmesher. The world-XY
    lattice adds only strictly interior points; boundary edges are not densified.
    Multiple loops, holes and non-waterline open seams are explicit refusals.
    """
    if isinstance(cell_size, (bool, np.bool_)):
        raise WaveFieldError('waterline cell size must be positive and finite')
    try:
        size = float(cell_size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WaveFieldError('waterline cell size must be positive and finite') from exc
    if not math.isfinite(size) or size <= 0:
        raise WaveFieldError('waterline cell size must be positive and finite')
    if type(max_panels) is not int or max_panels < 1:
        raise WaveFieldError('no mathematical lid panels fit the source limit')
    v, f = np.asarray(vertices), np.asarray(faces)
    uses = Counter(tuple(sorted((int(a), int(b)))) for face in f
                   for a, b in zip(face, np.roll(face, -1)) if a != b)
    if any(count > 2 for count in uses.values()):
        raise WaveFieldError('waterline hull is nonmanifold')
    adj = defaultdict(list)
    for (a, b), count in uses.items():
        if count == 1:
            adj[a].append(b)
            adj[b].append(a)
    if not adj or any(len(neighbours) != 2 for neighbours in adj.values()):
        raise WaveFieldError('waterline must form one closed, nonbranching loop')
    start = min(adj)
    loop, visited, previous, current = [], set(), None, start
    while current not in visited:
        visited.add(current)
        loop.append(current)
        following = min(n for n in adj[current] if n != previous)
        previous, current = current, following
    if current != start or len(loop) != len(adj):
        raise WaveFieldError('waterline lid supports one loop only; holes/multiple bodies are unsupported')
    if not np.all(v[loop, 2] == 0.):
        raise WaveFieldError('waterline boundary must already be exactly z=0; hull vertices are never moved')
    xy = np.asarray(v[loop, :2], dtype=float)
    spans = np.ptp(xy, axis=0)
    # Bound the lattice and a conservative pair-work inventory before allocation.
    ratios = spans / size
    if not np.isfinite(ratios).all() or np.any(ratios > 2 * max_panels + 2):
        raise WaveFieldError('waterline lattice exceeds bounded generation work')
    nx, ny = (math.ceil(float(r)) + 2 for r in ratios)
    count = nx * ny + len(loop)
    if count > 4 * max_panels + len(loop) or count * count * 64 > max_work_bytes:
        raise WaveFieldError('waterline lattice exceeds bounded generation work')
    try:
        import shapely
        from shapely.geometry import Polygon
        from anymesher.triangulation import triangulate_polygon
    except ImportError as exc:
        raise WaveFieldError('waterline lid generation requires the waterline-lid optional dependencies') from exc
    polygon = Polygon(xy)
    if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
        raise WaveFieldError('waterline must be a simple polygon with positive area')
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    xs = (math.floor(float(lo[0] / size)) + np.arange(nx) + .5) * size
    ys = (math.floor(float(lo[1] / size)) + np.arange(ny) + .5) * size
    grid = np.stack(np.meshgrid(xs, ys, indexing='ij'), axis=-1).reshape(-1, 2)
    grid = grid[shapely.contains_xy(polygon, grid[:, 0], grid[:, 1])]
    # For a triangulated simple polygon with no added boundary points: F=B+2I-2.
    expected_faces = len(loop) + 2 * len(grid) - 2
    if expected_faces > max_panels:
        raise WaveFieldError('hull plus fitted lid exceeds full source panel limit')
    try:
        result = triangulate_polygon(xy, interior_points=grid, backend='python')
    except Exception as exc:
        raise WaveFieldError(f'shared ANYmesher waterline triangulation failed: {exc}') from exc
    points = np.asarray(result.points, dtype=float)
    triangles = np.array(result.triangles, dtype=np.int64, copy=True)
    if (len(triangles) != expected_faces or len(triangles) > max_panels
            or not np.array_equal(points[:len(loop)], xy)):
        raise WaveFieldError('shared triangulator changed the waterline or panel inventory')
    p = points[triangles]
    cross = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
             - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    if not np.isfinite(points).all() or np.any(cross == 0):
        raise WaveFieldError('waterline triangulation contains degenerate/nonfinite panels')
    union = shapely.union_all([Polygon(poly) for poly in p])
    length = max(1., float(spans.max()))
    tol = 128 * np.finfo(float).eps
    if (polygon.symmetric_difference(union).area > tol * length**2
            or union.boundary.hausdorff_distance(polygon.boundary) > tol * length):
        raise WaveFieldError('fitted lid does not cover the exact waterline polygon')
    edge_counts = Counter(tuple(sorted((int(a), int(b)))) for face in triangles
                          for a, b in zip(face, np.roll(face, -1)))
    boundary = {edge for edge, n in edge_counts.items() if n == 1}
    expected = {tuple(sorted((i, (i + 1) % len(loop)))) for i in range(len(loop))}
    if boundary != expected or any(n not in (1, 2) for n in edge_counts.values()):
        raise WaveFieldError('fitted lid does not preserve every original boundary segment')
    triangles[cross > 0] = triangles[cross > 0][:, [0, 2, 1]]
    return np.column_stack([points, np.zeros(len(points))]), np.column_stack([triangles, triangles[:, -1]])
