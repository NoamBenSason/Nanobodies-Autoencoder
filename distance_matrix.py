import os
import sys
import numpy as np
import itertools
from Bio.PDB.PDBParser import PDBParser


def get_protein_residues(fn):
    parser = PDBParser()
    pdb_code = fn.split(".")[0]
    structure = parser.get_structure(pdb_code, fn)
    # take only amino acids elements
    residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
    return residues


def calc_distance_mat(file_name):
    protein_res = get_protein_residues(file_name)
    dims = len(protein_res)
    print(dims)
    distances_mat = np.empty((dims, dims))
    for row in range(dims):
        for col in range(dims):
            first_coord = protein_res[row]["CA"].get_coord()
            second_coord = protein_res[col]["CA"].get_coord()
            distances_mat[row, col] = np.linalg.norm(first_coord - second_coord)
    return distances_mat


if __name__ == '__main__':
    path = sys.argv[1]
    directory = os.fsencode(path)
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        print(calc_distance_mat(file_name))
