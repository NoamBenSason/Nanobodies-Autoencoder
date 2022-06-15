import subprocess as sp
import sys

renumber = "/cs/staff/dina/utils/srcs/renumber/renumber"
rmsd_prog = "/cs/staff/dina/utils/rmsd"

if __name__ == '__main__':
    pdb = sys.argv[1]
    # renumber ref chain if this is the first model
    sp.run(f"{renumber} ref.pdb > ref_renumber.pdb", shell=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)

    # calculate the total RMSD
    try:
        rmsd = float(sp.run(f"{rmsd_prog} -t ref_renumber.pdb model.pdb | tail -n1 ", shell=True, capture_output=True).stdout.strip())
    # rmsd program failed
    except ValueError:
        print(pdb, "error")
        valid = False
        continue

    # find sequences
    aa_seq =
    cdr1_seq =
    cdr2_seq =
    cdr3_seq =

    # get the cdrs locations according to IMGT numbering
    cdr1_start, cdr1_end = aa_seq.find(cdr1_seq)
    cdr2_start, cdr2_end = aa_seq.find(cdr2_seq)
    cdr3_start, cdr3_end = aa_seq.find(cdr3_seq)

    # change heavy numbering to Chothia
    cdr1_end -= 1
    cdr2_start += 1
    cdr2_end -= 1
    cdr3_start += 2

    # align to get rmsd of cdrs after aligning only the frame region...
    transformation = str(sp.run(f"{rmsd_prog} ref.pdb model -t | head -n1", shell=True, capture_output=True).stdout.strip().decode("utf-8"))

    # transform model to ref frame region
    sp.run(f"{transform} {transformation} < model.pdb > model_tr.pdb", shell=True, stdout=sp.DEVNULL,stderr=sp.DEVNULL)

    # get ref cdrs
    sp.run(f"{get_frag_chain} ref_renumber.pdb H {cdr1_start+1} {cdr1_end} > ref_cdr1.pdb", shell=True, stdout=sp.DEVNULL,stderr=sp.DEVNULL)
    sp.run(f"{get_frag_chain} ref_renumber.pdb H {cdr2_start+1} {cdr2_end} > ref_cdr2.pdb", shell=True, stdout=sp.DEVNULL,stderr=sp.DEVNULL)
    sp.run(f"{get_frag_chain} ref_renumber.pdb H {cdr3_start+1} {cdr3_end} > ref_cdr3.pdb", shell=True, stdout=sp.DEVNULL,stderr=sp.DEVNULL)

    # get ab model cdrs
    sp.run(f"{get_frag_chain} model_tr.pdb H {cdr1_start+1} {cdr1_end} > model_cdr1.pdb", shell=True, stdout=sp.DEVNULL,stderr=sp.DEVNULL)
    sp.run(f"{get_frag_chain} model_tr.pdb H {cdr2_start+1} {cdr2_end} > ram_cdr2.pdb", shell=True, stdout=sp.DEVNULL,stderr=sp.DEVNULL)
    sp.run(f"{get_frag_chain} model_tr.pdb H {cdr3_start+1} {cdr3_end} > ram_cdr3.pdb", shell=True, stdout=sp.DEVNULL,stderr=sp.DEVNULL)

    # calculate cdrs rmsd
    cdr1_rmsd = float(sp.run(f"{rmsd_prog} ref_cdr1.pdb model_cdr1.pdb | tail -n1 ", shell=True, capture_output=True).stdout.strip())
    cdr2_rmsd = float(sp.run(f"{rmsd_prog} ref_cdr2.pdb model_cdr2.pdb | tail -n1 ", shell=True, capture_output=True).stdout.strip())
    cdr3_rmsd = float(sp.run(f"{rmsd_prog} ref_cdr3.pdb model_cdr3.pdb | tail -n1 ", shell=True, capture_output=True).stdout.strip())

    print(cdr1_rmsd, cdr2_rmsd, cdr3_rmsd)


    # calculate frame rmsd
    # n = len(sequence)
    # n1, n2, n3 = (cdr1_end - cdr1_start ), (cdr2_end - cdr2_start ), (cdr3_end - cdr3_start )
    # m = n - n1 - n2 - n3
    # fr_rmsd = (((rmsd_all_fr*2)*n - (cdr1_rmsd2)*n1 - (cdr2_rmsd2)*n2 - (cdr3_rmsd2)*n3) / m) * 0.5