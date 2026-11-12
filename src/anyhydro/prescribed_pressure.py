"""Stored ANYhydro pressure jumps to the common prescribed-pressure pipeline.

No hydrodynamic calculation or qualification is performed. The initial adapter
is sheet-only: the canonical pressure_jump field contains no hull pressures.
"""
import json
import math
import numpy as np

from .orcawave_pressure import ImportedPressureBank

PrescribedPressureBank = ImportedPressureBank


def bank_from_anyhydro_pressure_jump(dataset, bodies, request, *, body_id, heading):
    """Select one body's stored total-excitation sheet pressures (Pa/m).

    Requires an already audited, saved excitation-source declaration and the
    complete original body/request inventory. Native phasors already use -iwt;
    only the oriented sheet reference normal is reversed, never the phasor.
    Original string IDs, dataset identity, coupling and qualification survive
    in provenance. Numeric panel IDs are portable surrogate identities.
    """
    from .backends import ImportedHydroDatasetBackend
    from .excitation_source import verify_excitation_source
    from .hashing import dataset_content_hash
    from anygeometry.panel_pressure_projection import validate_panel_geometry

    bodies=tuple(bodies)
    # This backend's solve method is strict identity-checked stored-data replay.
    stored=ImportedHydroDatasetBackend(dataset).solve(bodies,request)
    declaration=verify_excitation_source(stored,request,bodies)
    if 'pressure_jump' not in stored:
        raise ValueError('stored total-excitation pressure_jump is required')
    names=tuple(str(v) for v in stored.body.values)
    if type(body_id) is not str or body_id not in names:
        raise ValueError('explicit included original body ID required')
    if isinstance(heading,(bool,np.bool_)) or not np.isfinite(heading):
        raise ValueError('finite native heading required')
    headings=np.flatnonzero(np.remainder(stored.wave_direction.values,2*np.pi)==heading%(2*np.pi))
    if len(headings)!=1:raise ValueError('exact requested heading unavailable')
    columns={(str(s),int(f)):i for i,(s,f) in enumerate(zip(stored.sheet_panel_surface_id.values,stored.sheet_panel_face.values))}
    selected=next(b for b in bodies if b.body_id==body_id)
    polygons=[]; indices=[]; identities=[]; surfaces=[]
    for mesh in selected.attached_surfaces:
        if mesh.surface.surface_kind!='thin_sheet':continue
        if mesh.surface.primary_hydrodynamics!='potential_flow':
            raise ValueError('sheet pressure requires potential_flow ownership')
        for face_index,(face,size) in enumerate(zip(mesh.faces,mesh.face_sizes)):
            if size not in (3,4):raise ValueError('prescribed pressure supports triangular/quadrilateral sheet panels only')
            sid=mesh.surface.surface_id
            key=(sid,face_index)
            if key not in columns:raise ValueError('stored pressure panel identity missing')
            polygons.append(mesh.vertices[face[:size]][::-1])
            indices.append(columns[key]);surfaces.append(sid)
            raw_id=stored.sheet_panel.values[columns[key]]
            identities.append(dict(body_id=body_id,surface_id=sid,face_index=face_index,
                                   sheet_panel_id=raw_id.item() if isinstance(raw_id,np.generic) else raw_id,
                                   mesh_hash=mesh.mesh_hash))
    if not polygons:raise ValueError('selected body has no stored thin-sheet pressures')
    polygons,normals,centers,areas=validate_panel_geometry(polygons)
    vertices=[];faces=[]
    for polygon in polygons:
        face=list(range(len(vertices),len(vertices)+len(polygon)))
        if len(face)==3:face.append(face[-1])
        faces.append(face);vertices.extend(polygon)
    def decoded(value):return json.loads(value) if isinstance(value,str) else value
    provenance=dict(provider='ANYhydro',adapter='anyhydro_pressure_jump_v1',
        dataset_content_hash=dataset_content_hash(stored),selected_body_id=body_id,
        body_ids=list(names),panel_identities=identities,scope='selected_body_sheets_only',
        qualification_status=stored.attrs.get('qualification_status'),
        surface_identity=decoded(stored.attrs['surface_identity']),
        sheet_source_provenance=decoded(stored.attrs['sheet_source_provenance']),
        pressure_jump_contract=decoded(stored.attrs['pressure_jump_contract']),
        excitation_source_contract=declaration,
        conversion='canonical -iwt phasor unchanged; reference normal=-mesh minus-to-plus normal')
    bank=ImportedPressureBank(vertices=np.asarray(vertices),faces=np.asarray(faces),centers=centers,normals=normals,
        areas=areas,omegas=stored.omega.values,
        responses=stored.pressure_jump.values[:,int(headings[0]),indices],
        panel_ids=tuple(range(len(polygons))),surface_ids=tuple(surfaces),heading=float(heading%(2*np.pi)),
        composition='fixed_diffraction',body_id=names.index(body_id),
        water_depth=None if math.isinf(request.water_depth) else request.water_depth,rho=request.rho,gravity=request.gravity,
        provenance_json=json.dumps(provenance,sort_keys=True,allow_nan=False),panel_kinds=('dipole_jump',)*len(polygons))
    bank.validate_geometry()
    return bank
