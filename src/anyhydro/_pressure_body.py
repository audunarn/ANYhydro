"""Shared regular/bank mesh admission; default producer semantics preserved."""
import numpy as np

def prepare_body(cpt, vertices, faces, surface_ids, *, period, heading, water_depth, rho, max_panels):
    v = np.array(vertices, dtype=float, copy=True)
    raw_faces = np.asarray(faces)
    if raw_faces.dtype.kind not in 'iu' or raw_faces.dtype.kind == 'b':
        raise ValueError('integer face indices required')
    f = np.array(raw_faces, dtype=int, copy=True)
    labels = tuple(str(x) for x in surface_ids)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError('finite N by 3 vertices required')
    if f.ndim != 2 or f.shape[1] != 4 or len(f) != len(labels) or not 0 < len(f) <= min(600, max_panels):
        raise ValueError('one surface id per quad/degenerate triangle; panel limit 600')
    if f.min() < 0 or f.max() >= len(v) or v[:, 2].max() > 1e-9:
        raise ValueError('invalid mesh indices or hull above mean waterline')
    if not all(np.isfinite(x) for x in (period, heading, water_depth, rho)) or min(period, water_depth, rho) <= 0:
        raise ValueError('positive finite period, depth and density required')
    if v[:, 2].min() <= -water_depth:
        raise ValueError('source hull must lie strictly above the seabed')
    mesh = cpt.Mesh(vertices=v, faces=f, name='fixed-wet-hull')
    if np.any(mesh.faces_areas <= 0):
        raise ValueError('degenerate source panel')
    edges = {}
    for face in f:
        for first,last in zip(face,np.roll(face,-1)):
            if first == last:
                continue
            edges.setdefault(tuple(sorted((int(first),int(last)))),[]).append((int(first),int(last)))
    for key,uses in edges.items():
        if len(uses) == 1:
            if not np.allclose(v[list(key),2],0.,atol=1e-9,rtol=0.):
                raise ValueError('source mesh has open hull seams below the waterline')
        elif len(uses) != 2 or uses[0] != uses[1][::-1]:
            raise ValueError('source mesh is nonmanifold or inconsistently oriented')
    volume = np.sum(np.einsum('ij,ij->i',mesh.faces_centers,mesh.faces_normals)*mesh.faces_areas)/3.
    if volume <= 0:
        raise ValueError('source hull winding must enclose positive volume below z=0')
    body = cpt.FloatingBody(mesh=mesh, name='fixed-hull')
    body.add_all_rigid_body_dofs(rotation_center=(0., 0., 0.))
    return v, f, labels, mesh, body
