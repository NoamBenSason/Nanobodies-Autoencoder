import subprocess as sp
import sys
import pandas as pd

"""
this script is based on tomer's script, used to calc RMSD for CDR's
"""

renumber = "/cs/staff/dina/utils/srcs/renumber/renumber"
rmsd_prog = "/cs/staff/dina/utils/rmsd"
transform = "/cs/staff/dina/utils/pdb_trans"
get_frag_chain = "/cs/staff/dina/utils/get_frag_chain.Linux"


def calc(ref_pdb, predicted_pdb):
    """
    Calculates the RMSDs for cdr1, cdr2 and cdr3, given the reference pdb file
    and the pdb file of the prediction.
    :param ref_pdb: The reference pdb file.
    :param predicted_pdb: The pdb file of the prediction
    :return: the RMSDs for cdr1, cdr2 and cdr3, respectively
    """
    # renumber ref chain if this is the first model
    sp.run(f"{renumber} {ref_pdb} > ref_renumber.pdb", shell=True,
           stdout=sp.DEVNULL, stderr=sp.DEVNULL)

    pdb_file_name = ref_pdb + "_CDRS.csv"
    df = pd.read_csv(pdb_file_name)
    aa_seq = df.iloc[0, 1]
    cdr1_seq = df.iloc[0, 4]
    cdr2_seq = df.iloc[0, 5]
    cdr3_seq = df.iloc[0, 6]

    # conv to string
    aa_seq_str = str(aa_seq)
    cdr1_seq_str = str(cdr1_seq)
    cdr2_seq_str = str(cdr2_seq)
    cdr3_seq_str = str(cdr3_seq)

    # get the cdrs locations according to IMGT numbering
    cdr1_start = aa_seq_str.find(cdr1_seq_str)
    cdr1_end = cdr1_start + len(cdr1_seq_str)
    cdr2_start = aa_seq_str.find(cdr2_seq_str)
    cdr2_end = cdr2_start + len(cdr2_seq_str)
    cdr3_start = aa_seq_str.find(cdr3_seq_str)
    cdr3_end = cdr3_start + len(cdr3_seq_str)

    # change heavy numbering to Chothia
    cdr1_end -= 1
    cdr2_start += 1
    cdr2_end -= 1
    cdr3_start += 2

    # align to get rmsd of cdrs after aligning only the frame region...
    transformation = str(
        sp.run(f"{rmsd_prog} {ref_pdb} {predicted_pdb} -t | head -n1",
               shell=True,
               capture_output=True).stdout.strip().decode("utf-8"))

    # transform model to ref frame region
    sp.run(f"{transform} {transformation} < {predicted_pdb}> model_tr.pdb",
           shell=True, stdout=sp.DEVNULL,
           stderr=sp.DEVNULL)

    # get ref cdrs
    sp.run(
        f"{get_frag_chain} ref_renumber.pdb H {cdr1_start + 1} {cdr1_end} > ref_cdr1.pdb",
        shell=True,
        stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    sp.run(
        f"{get_frag_chain} ref_renumber.pdb H {cdr2_start + 1} {cdr2_end} > ref_cdr2.pdb",
        shell=True,
        stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    sp.run(
        f"{get_frag_chain} ref_renumber.pdb H {cdr3_start + 1} {cdr3_end} > ref_cdr3.pdb",
        shell=True,
        stdout=sp.DEVNULL, stderr=sp.DEVNULL)

    # get ab model cdrs
    sp.run(
        f"{get_frag_chain} model_tr.pdb H {cdr1_start + 1} {cdr1_end} > model_cdr1.pdb",
        shell=True,
        stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    sp.run(
        f"{get_frag_chain} model_tr.pdb H {cdr2_start + 1} {cdr2_end} > model_cdr2.pdb",
        shell=True,
        stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    sp.run(
        f"{get_frag_chain} model_tr.pdb H {cdr3_start + 1} {cdr3_end} > model_cdr3.pdb",
        shell=True,
        stdout=sp.DEVNULL, stderr=sp.DEVNULL)

    # calculate cdrs rmsd
    cdr1_rmsd = float(
        sp.run(f"{rmsd_prog} ref_cdr1.pdb model_cdr1.pdb | tail -n1 ",
               shell=True, capture_output=True).stdout.strip())
    cdr2_rmsd = float(
        sp.run(f"{rmsd_prog} ref_cdr2.pdb model_cdr2.pdb | tail -n1 ",
               shell=True, capture_output=True).stdout.strip())
    cdr3_rmsd = float(
        sp.run(f"{rmsd_prog} ref_cdr3.pdb model_cdr3.pdb | tail -n1 ",
               shell=True, capture_output=True).stdout.strip())

    sp.run("rm *.pdb", shell=True)

    return cdr1_rmsd, cdr2_rmsd, cdr3_rmsd
