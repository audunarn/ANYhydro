"""Fixed-body incident plus diffraction pressure, deliberately separate from RAOs.

The panel field is in Pa per metre wave amplitude, with exp(-i omega t).
Positive pressure acts opposite the supplied outward hull normal.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import numpy as np


@dataclass(frozen=True)
class RegularWavePressure:
    vertices: np.ndarray
    faces: np.ndarray
    surface_ids: tuple[str, ...]
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    pressure: np.ndarray
    omega: float
    heading: float
    water_depth: float
    rho: float
    source_id: str
    provider: str
    excitation_wrench: np.ndarray

    def validate(self):
        """Refuse changed payloads under a previously issued producer identity."""
        version=self.provider.split(';')[0].split()[-1]
        expected=hashlib.sha256(self.vertices.tobytes()+self.faces.tobytes()+self.pressure.tobytes()+self.normals.tobytes()+self.areas.tobytes()+repr((self.surface_ids,2*np.pi/self.omega,self.heading,self.water_depth,self.rho,version,'indirect')).encode()).hexdigest()
        if expected != self.source_id:
            raise ValueError('source pressure payload differs from producer identity')
        polygons=self.vertices[self.faces]
        av1=np.cross(polygons[:,1]-polygons[:,0],polygons[:,2]-polygons[:,0])/2.
        av2=np.cross(polygons[:,2]-polygons[:,0],polygons[:,3]-polygons[:,0])/2.
        a1=np.linalg.norm(av1,axis=1);a2=np.linalg.norm(av2,axis=1)
        centers=(a1[:,None]*polygons[:,:3].mean(axis=1)+a2[:,None]*polygons[:,[0,2,3]].mean(axis=1))/(a1+a2)[:,None]
        if not np.allclose(self.centers,centers,rtol=1e-12,atol=1e-10):
            raise ValueError('source centers differ from source panel geometry')
        force=-self.pressure[:,None]*self.areas[:,None]*self.normals
        wrench=np.r_[force.sum(axis=0),np.cross(self.centers,force).sum(axis=0)]
        if not np.allclose(wrench,self.excitation_wrench,rtol=1e-8,atol=1e-6):
            raise ValueError('source excitation differs from pressure integral')


def solve_regular_wave_pressure(vertices, faces, surface_ids, *, period=6.,
                                heading=0., water_depth=200., rho=1025.,
                                max_panels=600):
    """One bounded Capytaine 3.0 indirect diffraction solve; no radiation/motion.

    Mesh is an outward-oriented, waterline-terminated hull in SI coordinates.
    Caller supplies stable physical surface labels, retained in panel order.
    """
    import capytaine as cpt
    from capytaine.bem.airy_waves import airy_waves_pressure, froude_krylov_force
    if importlib.metadata.version('capytaine').split('.')[:2] != ['3', '0']:
        raise ValueError('regular pressure adapter requires Capytaine 3.0')
    from ._pressure_body import prepare_body
    v, f, labels, mesh, body = prepare_body(cpt, vertices, faces, surface_ids,
        period=period, heading=heading, water_depth=water_depth, rho=rho, max_panels=max_panels)
    problem = cpt.DiffractionProblem(body=body, omega=2*np.pi/period,
                                   wave_direction=heading, water_depth=water_depth, rho=rho)
    solver = cpt.BEMSolver()
    result = solver.solve(problem, method='indirect', keep_details=True, n_threads=1)
    total = np.asarray(result.pressure) + airy_waves_pressure(mesh.faces_centers, problem)
    fk = froude_krylov_force(problem)
    wrench = np.asarray([result.forces[k] + fk[k] for k in
                         ('Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw')])
    if not np.isfinite(total).all() or not np.isfinite(wrench).all():
        raise ValueError('non-finite BEM pressure/force')
    identity = hashlib.sha256(v.tobytes()+f.tobytes()+total.tobytes()+mesh.faces_normals.tobytes()+mesh.faces_areas.tobytes()+repr((labels, period, heading, water_depth, rho, cpt.__version__, 'indirect')).encode()).hexdigest()
    arrays = [v, f, mesh.faces_centers.copy(), mesh.faces_normals.copy(), mesh.faces_areas.copy(), total.copy(), wrench]
    for array in arrays:
        array.setflags(write=False)
    return RegularWavePressure(arrays[0], arrays[1], labels, arrays[2], arrays[3], arrays[4], arrays[5],
                               problem.omega, heading, water_depth, rho, identity,
                               f'capytaine {cpt.__version__}; {cpt.__file__}; indirect', arrays[6])
