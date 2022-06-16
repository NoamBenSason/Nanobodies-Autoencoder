import tensorflow as tf
import utils
import argparse
import os


def predict_single_sample(model, pdb_name, is_print):
    pdb_file_path = f"Ex4Data/{pdb_name}.pdb"
    test_sample = utils.generate_label(pdb_file_path)
    test_sample = test_sample[None, :]
    structure_out, seq_out2 = model.predict(test_sample)
    seq_str, real_seq_ind = utils.generate_ind(pdb_file_path)

    if is_print:
        print(seq_str)

    # saving a pdb file with all the predicted sequence (from the encoder )and structure (from the decoder)
    utils.matrix_to_pdb(seq_str, structure_out[0, :, :], f"Predicted/predicted_PDB/predict_{pdb_name}")

    # saving sequence to FASTA
    fasta_file = open(rf'Predicted/predicted_Fasta/{pdb_name}.Fasta', 'w+')
    fasta_file.write(f"> predicted sequence for: {pdb_name}\n")
    fasta_file.write(seq_str)
    fasta_file.close()


def main():
    # creating the directory for all the prediction files to be saved in
    parent_dir = "/cs/usr/noam_bs97/3D-Hackton-SeqDesign"
    path = os.path.join(parent_dir, "Predicted")
    if not os.path.exists("Predicted"):
        os.mkdir(path)
    path = os.path.join(f"{parent_dir}/Predicted","predicted_PDB")
    if not os.path.exists("Predicted/predicted_PDB"):
        os.mkdir(path)
    path = os.path.join(f"{parent_dir}/Predicted", "predicted_Fasta")
    if not os.path.exists("Predicted/predicted_Fasta"):
        os.mkdir(path)

    # parsing the arguments from the command line
    parser = argparse.ArgumentParser(description='Select model type and enter a PDB file')
    parser.add_argument('model_type', metavar='T', type=str, nargs=1,
                        help='a type of a model: N - normal, A - attention, MHA - multi headed attention')
    parser.add_argument('pdb_name', metavar='F', type=str, nargs=1,
                        help='a name of a pdb file (without suffix) with all the amino acid in the sequence unknown')
    parser.add_argument("--print_seq", action="store_true", help="a flag indicating if a sequence print is wanted")

    args = parser.parse_args()
    model_type = args.model_type[0]
    pdb_name = args.pdb_name[0]
    is_print = args.print_seq

    if model_type == "N":
        best_model = tf.keras.models.load_model("Best_models/efficient_original")
    elif model_type == "A":
        best_model = tf.keras.models.load_model("Best_models/prime_attntion")
    elif model_type == "MHA":
        best_model = tf.keras.models.load_model("Best_models/distinctive_multiheaded_attn")
    else:
        print("Model type does not exist")
        return

    predict_single_sample(best_model, pdb_name, is_print)


if __name__ == '__main__':
    main()
