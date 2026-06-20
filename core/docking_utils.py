import numpy as np
from Bio.PDB import PDBParser
"""  
    Compute docking box from full protein structure.

    Returns:
        center (x, y, z)
        size   (sx, sy, sz)
    """
def compute_box_from_pdb(pdb_path, padding=10.0):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    coords = [atom.get_coord() for atom in structure.get_atoms()]

    if len(coords) == 0:
        return (0.0, 0.0, 0.0), (30.0, 30.0, 30.0)

    coords = np.array(coords)

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)

    center = (min_coords + max_coords) / 2
    size = (max_coords - min_coords) + padding

    return tuple(center), tuple(size)


def compute_contact_box(protein_pdb, rna_pdb, cutoff=8.0, padding=6.0):
    """
    Focus docking box on protein regions close to RNA.
    """

    parser = PDBParser(QUIET=True)

    prot = parser.get_structure("prot", protein_pdb)
    rna = parser.get_structure("rna", rna_pdb)

    prot_coords = np.array([a.get_coord() for a in prot.get_atoms()])
    rna_coords = np.array([a.get_coord() for a in rna.get_atoms()])

    if len(prot_coords) == 0 or len(rna_coords) == 0:
        return None, None

    contact_points = []

    for p in prot_coords:
        dists = np.linalg.norm(rna_coords - p, axis=1)
        if np.min(dists) < cutoff:
            contact_points.append(p)

    if not contact_points:
        return None, None

    coords = np.array(contact_points)

    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)

    center = (min_c + max_c) / 2
    size = (max_c - min_c) + padding

    return tuple(center), tuple(size)

