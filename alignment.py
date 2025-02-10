
from Bio.motifs import Motif, Alignment
from Bio import motifs
from Bio.Seq import Seq as BioSeq
from lasagna_lib import LASAGNA_alignment
from tqdm import tqdm
import pandas
import logging
from Bio import SeqIO
from lasagna_lib import LASAGNA_alignment
from lasagna_lib import ComputeCounts
from lasagna_lib import ComputeIC
from lasagna_lib import ComputeCoverage
from lasagna_lib import SuggestSize
import numpy as np
from weblogo import *
import csv
import json
import decimal as dec
import os

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("logfile_alignment_m.log", mode='w'),  # Log to file
        logging.StreamHandler()  # Log to console (stdout)
    ]
)

def round_column(column, N = 96):
    '''
    Applies the "Largest Remainder Method" to round scaled frequencies - i.e.
    f*N where f is a frequency and N is a scalar - from a column of a scaled
    PWM, so that the sum remains N (i.e., the sum of the frequencies remains 1).
    For N=100, this is a method to deal with percentages, rounding them to
    integers in a way that minimizes the errors.
         column:  list of (four) frequencies, for the four nucleotides
         N:  desired total number of virtual instances for FLEMINGO. The smallest
             possible frequency will be 1/N (zeros are not allowed)
    '''
    # Frequencies are scaled to become (almost) counts
    column = column * N
    # Truncation: counts must be integers
    truncated = np.floor(column).astype(int)
    # Order: from most to least penalized count, i.e., sorted based on how big
    # was the effect of the truncation
    order = np.flip(np.argsort(column - truncated))
    # How many counts we miss (due to truncation) to reach N
    remainder = N - sum(truncated)
    # Distribute the `remainder` by prioritizing values that were most
    # penalized by the truncation (i.e., following `order`)
    for i in range(remainder):
        truncated[order[i]] += 1
    # Add +1 to each frequency it's 1 and the highest is N-3
    for i in range(4):
        truncated[i] += 1
    prob_old = truncated/(N+4)
    # To avoid extra decimals due to floating point errors:
    # Define the number of decimal digits actually required.
    # All frequencies are multiples of 1/N
    smallest_freq = dec.Decimal('1') / dec.Decimal(str(N+4))
    # Number of decimals in the smallest frequency
    no_decimals = len(str(smallest_freq).split(".")[1])
    # No frequency value needs more decimal digits than  smallest_freq.
    # Therefore we can round according to  no_decimals
    prob_new = np.array([round(num, no_decimals) for num in prob_old])
    return prob_new

def get_alignment_offset(motif, other):
    '''
    Determines the optimal alignment of two motifs by maximizing the information content (IC) in the aligned regions.
    Parameters
    ----------
    motif, other: Motif objects
        The two motifs of interest.
    
    Returns
    -------
    offsets: int
        The offset that results in the maximum IC in the alignment. 
    '''
    max_ic = float('-inf')
    for offset in range(-len(motif) + 1, len(other)): # Loop over a range of possible offset values
        # Calculate the IC for each alignment at each offset
        if offset < 0:
            ic = ic_at(motif, other, -offset)
        else:
            ic = ic_at(other, motif, offset)

        # Updates the maxium offset
        if ic > max_ic:
            max_ic = ic
            max_offset = offset
    return max_offset

def ic_at(motif, other, offset):
    '''
    Calculates the information content, IC, for a specific alignment. The approach makes a temporary motif object containing the overlapping sequences in the alignment and taking the average of the pssm.
    Parameters
    ----------
    motif, other: Motif objects
        The motifs of interest
    offset: int
        The offset value that results in the alignment of interest. 
    '''

    #Pull the sequences containined in the aligned region of the motifs from each of the motif instances. 
    alignment_len = min(len(motif)-offset, len(other))
    motif_seqs = [site[offset:alignment_len+offset] for site in motif.alignment.sequences]
    other_seqs = [site[:alignment_len] for site in other.alignment.sequences]

    # Create the motif and compute the IC
    amotif = Motif(alignment=Alignment(motif_seqs+other_seqs))
    amotif.pseudocounts = dict(A=0.25, C=0.25, G=0.25, T=0.25)

    #print('Motif Seqs: ' , motif_seqs)
    #print('Other Seqs: ' , other_seqs)
    #print('Offset ', offset)
    #print('IC: ' , amotif.pssm.mean(), '\n\n')

    return amotif.pssm.mean()

def createMotifAligned(motif, other):
    """
    Given 2 motifs with determinated aligned sites, creates a new motif that is the aligned combination of the sites of both initial motifs.
    Returns:
        - non_motif: A list of the aligned combination of the sites of both initial motifs.
        - left: Number of gaps on the left in the alignment in comparison with the original sequence.
        - right: Number of gaps on the right in the alignment in comparison with the original sequence.
        - inner: Number of gaps between recognizers.
        - og_offset: Offset with the original sign.
    """
    # Convert motif and other into Seq objects
    instances1=[]
    instances2=[]
    
    for seq in motif:
        instances1.append(BioSeq(seq))
        
    for seq in other:
        instances2.append(BioSeq(seq))
        
    motif=motifs.create(instances1)
    other=motifs.create(instances2)

    # Determine the optimal alignment offset
    offset=get_alignment_offset(motif, other)

    # Adjust the offset and swap motifs if necessary
    non_motif=[]
    og_offset = offset
    if offset<0:
        offset=-offset
    else:
        tmp=motif
        motif=other
        other=tmp
    # Align sequences based on the offset
    alignment_len = min(len(motif)-offset, len(other)) # Length of the overlapping region between the two motifs
    motif_seqs = [site[offset:alignment_len+offset] for site in motif.alignment.sequences] # Extract the aligned portion from the first motif
    other_seqs = [site[:alignment_len] for site in other.alignment.sequences] # Extract the aligned portion from the second motif
    amotif = Motif(alignment=Alignment(motif_seqs+other_seqs)) # Create the combined motif
    left = 0
    right = 0
    inner = 0
    if og_offset != 0:
        dif_motif = len(motif.alignment.sequences[0]) - len(motif_seqs[0])
        dif_other = len(other.alignment.sequences[0]) - len(other_seqs[0])
        dashes = "-" * dif_motif
        dashes2 = "-" * dif_other
    
    # Determine alignment based on symmetry
        if symmetry == "DIRECT-REPEAT":
            left, right, inner = (dif_motif, dif_other, 0) if og_offset < 0 else (0, dif_other, dif_motif)
        elif symmetry == "INVERTED-REPEAT":
            left, right, inner = (dif_motif, 0, dif_other) if og_offset < 0 else (0, dif_motif, dif_other)
    
    # Add gaps to sequences
        motif_seqs_gap = [BioSeq(str(seq) + dashes2) for seq in motif.alignment.sequences]
        other_seqs_gap = [BioSeq(dashes + str(seq)) for seq in other.alignment.sequences]
    
    # Create the final motif with gaps
        amotif_gap = Motif(alignment=Alignment(motif_seqs_gap + other_seqs_gap))
    else:
        amotif_gap = amotif
        motif_seqs_gap = motif_seqs
        other_seqs_gap = other_seqs
    if og_offset > 0:
        tmp = other_seqs_gap
        other_seqs_gap = motif_seqs_gap
        motif_seqs_gap = tmp
        
    # Create the logo
    create_logo(amotif_gap.alignment.sequences, "Gapped_logos", f"{tf}{mod}.png")
    create_logo(motif_seqs_gap, "Gapped_logos1", f"{tf}{mod}.png")
    create_logo(other_seqs_gap, "Gapped_logos2", f"{tf}{mod}.png")
    for seq in amotif.alignment.sequences: # Convert the motif back to string format
        non_motif.append(str(seq))
    return non_motif, left, right, inner, og_offset
    
def alignSites(sites):
    """
    Given the sites of all motifs of the input, it aligns them
    Returns:
        - sites: Aligned motifs
        - left: Number of gaps on the left in the alignment in comparison with the original sequence
        - right: Number of gaps on the right in the alignment in comparison with the original sequence
        - inner: Number of gaps between recognizers
        - og_offset: Offset with the original sign
    """
    #Count how many sites has every motif
    quantity_of_sequences=[]
    for seq in sites:
        quantity_of_sequences.append(len(seq))
    #Loop inicialization, align the first motif if needed. When the lenght is not the same it uses LASAGNA as a tool for aligning  
    it = iter(sites[0])
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        sites[0]=LASAGNA_alignment(sites[0])
    #Loop to align all motifs
    for i in tqdm(range(len(sites)-1)):
        it = iter(sites[i+1])
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            sites[i+1]=LASAGNA_alignment(sites[i+1])
        sites[i+1], left, right, inner, og_offset=createMotifAligned(sites[i],sites[i+1])
    #now that we have all the sites of all motifs aligned in sites[-1] we have to divide them to storage them in the respective motif
    i=0
    old_quantity=0
    for quantity in quantity_of_sequences:
        sites[i]=sites[-1][old_quantity:old_quantity+quantity]
        i=i+1
        old_quantity=old_quantity+quantity
    return sites, left, right, inner, og_offset

def seq_split(sequences, symmetry):
    '''
    Split a list of sequences into halves and process them based on the specified symmetry.
    Args:
         sequences:  List of string or Bio.Seq objects for splitting in half.
         symmetry:  symmetry of the factor, either "INVERTED-REPEAT" or "DIRECT-REPEAT"
    Returns:
        Three lists containing:
        - submotifs: A flat list of first and second halves as strings.
        - submotifs_t: A list of tuples, each containing (first_half, second_half).
        - submotifs12: A list containing two lists: first halves and second halves.
    '''
    submotifs_t = []
    submotifs1 = []
    submotifs2 = []
    submotifs12 = []
    # Get the middle position
    for sequence in sequences:
        half = len(sequence) // 2 # División entera
    # Divide the sequence
        first_half = sequence[:half]  # Elements from the beginning until half (not included)
        second_half = sequence[half:]  # Elements from the half until the end
        if symmetry == "INVERTED-REPEAT":
            second_half = second_half.reverse_complement() # Reverse complement the second submotif
    # Append the sequences to the lists
        submotifs1.append(str(first_half))
        submotifs2.append(str(second_half))
        submotifs_t.append((str(first_half), str(second_half)))
    submotifs = [item for pair in submotifs_t for item in pair]  # Flattened list
    submotifs12 = [submotifs1, submotifs2]
    return submotifs, submotifs_t, submotifs12

def suggest_trimming(trimmed_sequences, left, right, inner, symmetry):
    '''
    Suggests trimming based on Information Content (IC) and Coverage.
    Args:
        - trimmed_sequences: List of sequences to be trimmed.
        - left: Number of gaps on the left in the alignment in comparison with the original sequence.
        - right: Number of gaps on the right in the alignment in comparison with the original sequence.
        - inner: Number of gaps between recognizers.
        - symmetry: Type of symmetry, either "DIRECT-REPEAT" or "INVERTED-REPEAT".
    Returns:
        Returns:
        - trimmed_sequences: List of sequences trimmed.
        - left: Number of gaps on the left in the alignment in comparison with the original sequence.
        - right: Number of gaps on the right in the alignment in comparison with the original sequence.
        - inner: Number of gaps between recognizers.
    '''
    # Initial computation
    counts = ComputeCounts(trimmed_sequences, 0)
    Ic = ComputeIC(trimmed_sequences, 0, counts=counts[1])[0]
    Coverage = ComputeCoverage(counts=counts[1], nSites=len(trimmed_sequences))
    Ic_threshold = np.mean(Ic)
    # Suggest trimming, trimming and computation
    trimmedL, cntL, cntR = SuggestSize(Ic, Coverage, ICThres=Ic_threshold)
    if symmetry == "INVERTED-REPEAT":
        left = left + cntL
        right = right + cntL
        inner = inner + (cntR * 2)
    elif symmetry == "DIRECT-REPEAT":
        left = left + cntL
        right = right + cntR
        inner = inner + cntL + cntR
    trimmed_sequences = [lst[cntL:len(lst) if cntR == 0 else -cntR] for lst in trimmed_sequences]
    trimmed_sequences_gap = []
    # Add "-" depending on the gaps given by left, right, inner
    for i in range(0, len(trimmed_sequences), 2):
        first_element = trimmed_sequences[i]
        if symmetry == "INVERTED-REPEAT":
            second_element = str(BioSeq(trimmed_sequences[i + 1]).reverse_complement())
            trimmed_sequences_gap.append(f"{left*"-"}{first_element}{inner*"-"}{second_element}{right*"-"}")
        else:
            second_element = trimmed_sequences[i + 1]
            trimmed_sequences_gap.append(f"{left*"-"}{first_element}{inner*"-"}{second_element}{right*"-"}")
    # Create the logo
    create_logo(trimmed_sequences_gap, "Trimmed_gap_logos", f"{tf}{mod}.png")
    counts = ComputeCounts(trimmed_sequences, 0)
    Ic = ComputeIC(trimmed_sequences, 0, counts=counts[1])[0]
    average_Ic = np.mean(Ic)
    # Log details
    logging.info(f'Information content: {str(Ic).replace("\n", " ")}')
    logging.info(f'Coverage: {str(Coverage)}')
    logging.info(f'IC threshold: {str(Ic_threshold)}')
    logging.info(f'Left trim: {str(cntL)}')
    logging.info(f'Right trim: {str(cntR)}')
    logging.info(f'Final length: {str(trimmedL)}')
    logging.info(f'Average IC after trimming: {str(average_Ic)}')
    # Check for empty sequences
    if all(element == "" for element in trimmed_sequences):
        logging.info("WARNING: All elements are empty strings. Adjust IC/Coverage threshold")
    # Write information in the csv
    spamwriter.writerow([tf, str(cntL), str(cntR), str(trimmedL), str(average_Ic)])
    return trimmed_sequences, inner, left, right

def write_output(merged_motif, out_folder, out_file, symmetry = "DIRECT-REPEAT"):
    """
    Writes the merged motif to an output file, optionally handling inverted repeats.

    Args:
        merged_motif: List of sequences to write.
        out_folder: Path to the output folder.
        out_file: Name of the output file.
        symmetry: Type of symmetry, either "DIRECT-REPEAT" or "INVERTED-REPEAT".
    """
    # Ensure the output file exists
    os.makedirs(out_folder, exist_ok=True)
    # Open the file for writing in the output folder
    if symmetry == "INVERTED-REPEAT":
        merged_motif = [str(BioSeq(seq).reverse_complement()) for seq in merged_motif]
    # Create output file
    output_file = os.path.join(out_folder, out_file)
    prefix = out_file.split("_")[0] + "_"  # Extract prefix before the first underscore
    with open(output_file, "w") as ofile:
        for i, seq in enumerate(merged_motif):
            ofile.write(f">{prefix}{i}\n{seq}\n")

def write_json(merged_motif, symmetry, inner, left, right, output_folder="json_files"):
    """
    Writes a JSON file containing PSSM (Position-Specific Scoring Matrix) data for motifs.

    Args:
        - merged_motif: List of sequences for the motif.
        - symmetry: Type of symmetry, either "INVERTED-REPEAT" or "DIRECT-REPEAT".
        - tf: Transcription factor name for the output file.
        - output_folder: Folder to save the JSON file (default is "json_files").
        - left: Number of gaps on the left in the alignment in comparison with the original sequence.
        - right: Number of gaps on the right in the alignment in comparison with the original sequence.
        - inner: Number of gaps between recognizers.
    """
    # Create motifs for the merged motif and its complementary sequences if needed
    motif_seqs_motif = motifs.create([BioSeq(seq) for seq in merged_motif])
    if symmetry == "INVERTED-REPEAT":
        other_seqs_motif = motifs.create([BioSeq(seq).reverse_complement() for seq in merged_motif])
    else:
        other_seqs_motif = motifs.create(merged_motif)
    trimmed_rev_sequences_gap = []
    for i in range(0, len(motif_seqs_motif.alignment.sequences)):
        first_element = motif_seqs_motif.alignment.sequences[i]
        second_element = other_seqs_motif.alignment.sequences[i]
        trimmed_rev_sequences_gap.append(f"{left*"-"}{first_element}{inner*"-"}{second_element}{right*"-"}")
    create_logo(trimmed_rev_sequences_gap, "Trimmed_rev_gap_logos", f"{tf}{mod}.png")
    # Convert motifs.pwm to dictionaries
    motif_dict = zip(*motif_seqs_motif.pwm.values())
    other_dict = zip(*other_seqs_motif.pwm.values())
    converted_motif_dict = [
    {key.lower(): values[i] for i, key in enumerate(motif_seqs_motif.pwm)}
    for values in motif_dict
    ]
    converted_other_dict = [
    {key.lower(): values[i] for i, key in enumerate(other_seqs_motif.pwm)}
    for values in other_dict
    ]
    first_recognizer = []
    # Round the values
    for dicts in converted_motif_dict:
        values = np.array(list(dicts.values()))
        rounded_values = round_column(values)
        rounded_dict = {key: rounded_values[i] for i, key in enumerate(dicts)}
        first_recognizer.append(rounded_dict)
    second_recognizer = []
    for dicts in converted_other_dict:
        values = np.array(list(dicts.values()))
        rounded_values = round_column(values)
        rounded_dict = {key: rounded_values[i] for i, key in enumerate(dicts)}
        second_recognizer.append(rounded_dict)
    # Create the file to be imported
    json_import = [[
        {"objectType":"pssm","pwm":first_recognizer},
        {"objectType":"connector","mu":float(inner),"sigma":0.0},
        {"objectType":"pssm","pwm":second_recognizer}
        ]]
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    output_file_json = os.path.join(output_folder, f"{tf}.json")
    with open(output_file_json, "w") as file:
        json.dump(json_import, file, indent=2)

def create_logo(sequences, folder, file_name, color_scheme="color_classic"):
    """
    Creates a sequence logo and saves it to a file.

    Args:
        - sequences: List of sequences for the logo.
        - folder: Output folder for the logo file.
        - file_name: Name of the logo file.
        - color_scheme: Color scheme for the logo.
    """
    output_path = os.path.join("logos", folder)
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, file_name)
    if not os.path.exists(output_path):
        motif = motifs.create([BioSeq(seq) for seq in sequences])
        motif.weblogo(output_path, color_scheme=color_scheme)

def process_motif(file, symmetry, sub_list, out_file, mod):
    """
    Processes a motif from a FASTA file, creating sequence logos and splitting motifs as needed.

    Args:
        - file: Input FASTA file containing sequences.
        - symmetry: Symmetry type, either "DIRECT-REPEAT" or "INVERTED-REPEAT".
        - tf: Transcription factor name for naming outputs.
        - sub_list: List to append the output file name to.
        - out_file: Name of the output file.
        - mod: Add a modifier tag if the files have been modified.

    Returns:
        Tuple of processed alignments and output file name.
    """
    # Parse sequences from the FASTA file
    with open(file) as fasta_file:
        sequences = [s.seq for s in SeqIO.parse(fasta_file, "fasta")]

        # Create the initial logo
        create_logo(sequences, "Original_logos", f"{tf}{mod}.png")
        logging.info(f"Original sequences: {str(sequences)}")

        if len(sequences[0]) < 7:
            # If motif is shorter than 7 bp, assume it's a half-site and skip further processing
            sub_list.append(out_file)
            sequences = [str(seq) for seq in sequences]
            logging.info('No splitting required. Length is less than 7bp.')
            return sequences, out_file
        
        else:
            # Split the motif into two halves
            split_sequences, split_sequences_t, split_12 = seq_split(sequences, symmetry)
            logging.info(f'Split sequences: {str(split_sequences_t)}')

            # Align the submotifs using the alignSites from cgb3 function
            alignments, left, right, inner, og_offset= alignSites(split_12)
            if og_offset >= 0:
                alignments = alignments[::-1]
            alignments = [item for pair in zip(*alignments) for item in pair]
            
            # Create the aligned logo
            create_logo(alignments, "Aligned_logos_m", f"{tf}{mod}.png")
            logging.info(f'Aligned sequences: {str(alignments)}')

            # Trim alignments and create final trimmed logo
            alignments_trim, inner, left, right = suggest_trimming(alignments, left, right, inner, symmetry)
            create_logo(alignments_trim, "Aligned_logos_m_trimmed", f"{tf}{mod}.png")
            logging.info(f'Final Trimmed Alignment: {str(alignments_trim)}')
            sub_list.append(out_file)
            return alignments_trim, out_file, inner, left, right  # Return the merged submotif

import ssl

# This is to avoid not connecting to weblogo
ssl._create_default_https_context = ssl._create_stdlib_context
excel = "tabla_TFs.xlsx"
excel_input = pandas.read_excel(excel, sheet_name = 0)
motifs_list = excel_input["Motif_file"].tolist()
repeat_list = excel_input["Symmetry"].tolist()
submotif_list = []
input_folder = "Binding_motifs_TFs_coli"
csv_output = "Alignment_m.csv"
with open(csv_output, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for file, symmetry in zip(motifs_list, repeat_list):  
        if pandas.isna(file):
            submotif_list.append(file)
            continue
        elif symmetry == "ASYMMETRIC":
            submotif_list.append("")
            continue
        mod = ""
        tf = file.split("_")[0]
        if "_mod" in file:
            mod = "_mod"
        logging.info(f'Processing TF: {tf} with Symmetry: {symmetry}')
        file_path = os.path.join(input_folder, file)
        out_file = f"{tf}_submotif.fas"
        merged_motif, out_file, inner, left, right = process_motif(file_path, symmetry, submotif_list, out_file, mod)
        write_output(merged_motif, "Binding_submotif_1", out_file)
        write_output(merged_motif, "Binding_submotif_2", out_file, symmetry)
        write_json(merged_motif, symmetry, inner, left, right)
excel_input["Submotif_files"] = submotif_list
excel_input.to_excel("tabla_TFs_submotif.xlsx", index=False)