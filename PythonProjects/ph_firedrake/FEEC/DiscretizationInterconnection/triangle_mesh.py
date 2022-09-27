from pyop2.mpi import COMM_WORLD
from firedrake.cython import dmcommon
from firedrake import mesh


def create_reference_triangle(h, comm=COMM_WORLD, name="ReferenceTriangle", \
                            reorder=None, distribution_parameters=None):

    """
    The boundary label are
    1 : y = 0
    2 : y = h - x
    3 : x = 0
    Counterclockwise orientation, normal outwards
    """

    coords = [[0., 0.], [h, 0.], [0., h]]
    cells = [[0, 1, 2]]
    plex = mesh._from_cell_list(2, cells, coords, comm)

    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    # coords = plex.getCoordinates()
    # print(coords)
    # coord_sec = plex.getCoordinateSection()
    boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()

    kk = 1
    for face in boundary_faces:
        # print(face)
        # face_coords = plex.vecGetClosure(coord_sec, coords, face)
        # print(face_coords)

        plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, kk)
        kk = kk + 1
    plex.removeLabel("boundary_faces")
    return mesh.Mesh(plex, reorder=reorder, distribution_parameters=distribution_parameters, name=name)
