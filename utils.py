# delete this cell if working on Pycharm
# !pip install Bio

from Bio.PDB import *
import numpy as np
import os
from tqdm import tqdm

NB_MAX_LENGTH = 140
AA_DICT = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7,
           "K": 8, "L": 9, "M": 10, "N": 11,
           "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "W": 17, "Y": 18,
           "V": 19, "X": 20, "-": 21}
FEATURE_NUM = len(AA_DICT)
BACKBONE_ATOMS = ["N", "CA", "C", "O", "CB"]
BACKBONE_ATOMS_DICT = {"N": [0, 1, 2], "CA": [3, 4, 5], "C": [6, 7, 8],
                       "O": [9, 10, 11],
                       "CB": [12, 13, 14]}
ROW_IND = np.arange(0, NB_MAX_LENGTH)
OUTPUT_SIZE = len(BACKBONE_ATOMS) * 3
NB_CHAIN_ID = "H"

def get_seq_aa(pdb_file, chain_id):
    """
    returns the sequence (String) and a list of all the aa residue objects of the given protein chain.
    :param pdb_file: path to a pdb file
    :param chain_id: chain letter (char)
    :return: sequence, [aa objects]
    """
    # load model
    chain = PDBParser(QUIET=True).get_structure(pdb_file, pdb_file)[0][chain_id]

    aa_residues = []
    seq = ""

    for residue in chain.get_residues():
        aa = residue.get_resname()
        if not is_aa(aa) or not residue.has_id('CA'):  # Not amino acid
            continue
        elif aa == "UNK":  # unkown amino acid
            seq += "X"
        else:
            seq += Polypeptide.three_to_one(residue.get_resname())
        aa_residues.append(residue)

    return seq, aa_residues

def generate_input(pdb_file):
    """
    receives a pdb file and returns its sequence in a one-hot encoding matrix (each row is an aa in the sequence, and
    each column represents a different aa out of the 20 aa + 2 special columns).
    :param pdb_file: path to a pdb file (nanobody, heavy chain has id 'H')
    :return: numpy array of shape (NB_MAX_LENGTH, FEATURE_NUM)
    """

    # get seq and aa residues
    seq, _ = get_seq_aa(pdb_file, NB_CHAIN_ID)
    if len(seq) > NB_MAX_LENGTH:
        seq = seq[:NB_MAX_LENGTH]
    if len(seq) < NB_MAX_LENGTH:
        seq = seq.ljust(NB_MAX_LENGTH, '-')
    indices = [AA_DICT[c] for c in seq]
    matrix = np.zeros((NB_MAX_LENGTH, FEATURE_NUM))
    matrix[ROW_IND, indices] = 1
    return matrix


def generate_ind(pdb_file):
    """
    receives a pdb file and returns its sequence and the ind list representing the matrix
    :param pdb_file: path to a pdb file (nanobody, heavy chain has id 'H')
    :return: sequence, ind list representing the matrix
    """

    # get seq and aa residues
    seq_r, _ = get_seq_aa(pdb_file, NB_CHAIN_ID)
    seq = seq_r
    if len(seq) > NB_MAX_LENGTH:
        seq = seq[:NB_MAX_LENGTH]
    if len(seq) < NB_MAX_LENGTH:
        seq = seq.ljust(NB_MAX_LENGTH, '-')
    return seq_r, [AA_DICT[c] for c in seq]


def generate_label(pdb_file):
    """
    receives a pdb file and returns its backbone + CB coordinates.
    :param pdb_file: path to a pdb file (nanobody, heavy chain has id 'H') already alingned to a reference nanobody.
    :return: numpy array of shape (CDR_MAX_LENGTH, OUTPUT_SIZE).
    """
    # get seq and aa residues
    seq, aa_residues = get_seq_aa(pdb_file, NB_CHAIN_ID)
    if len(aa_residues) > 140:
        aa_residues = aa_residues[:140]

    matrix = np.zeros((NB_MAX_LENGTH, OUTPUT_SIZE))

    for i, r in enumerate(aa_residues):
        for part, indices in BACKBONE_ATOMS_DICT.items():
            if part == "CB" and r.resname == "GLY":
                continue
            matrix[i,indices]=r[part].coord
    return matrix

def matrix_to_pdb(seq, coord_matrix, pdb_name):
    """
    Receives a sequence (String) and the output matrix of the neural network (coord_matrix, numpy array)
    and creates from them a PDB file named pdb_name.pdb.
    :param seq: protein sequence (String), with no padding
    :param coord_matrix: output np array of the nanobody neural network, shape = (NB_MAX_LENGTH, OUTPUT_SIZE)
    :param pdb_name: name of the output PDB file (String)
    """
    ATOM_LINE = "ATOM{}{}  {}{}{} {}{}{}{}{:.3f}{}{:.3f}{}{:.3f}  1.00{}{:.2f}           {}\n"
    END_LINE = "END\n"
    k = 1
    with open(f"{pdb_name}.pdb", "w") as pdb_file:
        for i, aa in enumerate(seq):
            third_space = (4 - len(str(i))) * " "
            for j, atom in enumerate(BACKBONE_ATOMS):
                if not (aa == "G" and atom == "CB"):  # GLY lacks CB atom
                    x, y, z = coord_matrix[i][3 * j], coord_matrix[i][
                        3 * j + 1], coord_matrix[i][3 * j + 2]
                    b_factor = 0.00
                    first_space = (7 - len(str(k))) * " "
                    second_space = (4 - len(atom)) * " "
                    forth_space = (12 - len("{:.3f}".format(x))) * " "
                    fifth_space = (8 - len("{:.3f}".format(y))) * " "
                    sixth_space = (8 - len("{:.3f}".format(z))) * " "
                    seventh_space = (6 - len("{:.2f}".format(b_factor))) * " "

                    pdb_file.write(
                        ATOM_LINE.format(first_space, k, atom, second_space,
                                         Polypeptide.one_to_three(aa), "H",
                                         third_space,
                                         i, forth_space, x, fifth_space, y,
                                         sixth_space, z, seventh_space,
                                         b_factor, atom[0]))
                    k += 1

        pdb_file.write(END_LINE)

if __name__ == '__main__':

    #  you can make all the data for the network in this section.
    # you can save the matrices to your drive and load them in your google colab file later.

    input_matrix = []
    labels_matrix = []
    data_path = "Ex4Data"

    for pdb in tqdm(os.listdir(data_path)):
        nb_one_hot = generate_input(os.path.join(data_path, pdb))
        nb_xyz = generate_label(os.path.join(data_path, pdb))

        input_matrix.append(nb_one_hot)
        labels_matrix.append(nb_xyz)

    save_path = "Ex4Files"

    np.save(f"train_input.npy", np.array(input_matrix))
    np.save(f"train_labels.npy", np.array(labels_matrix))

    print(f"Number of samples: {len(input_matrix)}")