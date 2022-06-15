import sys
import numpy as np
import itertools
from Bio.PDB.PDBParser import PDBParser

def get_protein_residues(file_name):
    parser = PDBParser()
    pdb_code = file_name.split(".")[0]
    structure = parser.get_structure(pdb_code, file_name)
    # take only amino acids elements
    residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
    return residues


def calc_distance_mat(file_name):
    protein_res = get_protein_residues(file_name)
    dims = len(protein_res)
    distances_mat = np.empty((dims, dims))
    for row in range(dims):
        for col in range(dims):
            first_coord = protein_res[row]["CA"].get_coord()
            second_coord = protein_res[col]["CA"].get_coord()
            distances[row, col] =  np.linalg.norm(first_coord - second_coord)
    return distances_mat


if __name__ == '__main__':
    path = sys.argv[1]
    for file_name in pdb_path:
        calc_distance_mat(file_name)