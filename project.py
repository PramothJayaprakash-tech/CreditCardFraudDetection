import pandas as pd

# FAO/WHO reference patterns (in mg of amino acid per gram of protein)
reference_patterns = {
    'Infants': {
        'Histidine': 18,
        'Isoleucine': 25,
        'Leucine': 55,
        'Lysine': 51,
        'Methionine + Cysteine': 25,
        'Phenylalanine + Tyrosine': 47,
        'Threonine': 27,
        'Tryptophan': 7,
        'Valine': 32
    },
    'Adolescents': {
        'Histidine': 20,
        'Isoleucine': 30,
        'Leucine': 55,
        'Lysine': 50,
        'Methionine + Cysteine': 25,
        'Phenylalanine + Tyrosine': 47,
        'Threonine': 30,
        'Tryptophan': 10,
        'Valine': 30
    },
    'Adults': {
        'Histidine': 14,
        'Isoleucine': 20,
        'Leucine': 39,
        'Lysine': 45,
        'Methionine + Cysteine': 15,
        'Phenylalanine + Tyrosine': 34,
        'Threonine': 15,
        'Tryptophan': 4,
        'Valine': 26
    }
}

# Load the amino acid profile from the provided Excel sheet
file_path = r'C:\Users\hp\Desktop\project\Pulse database for projects 2 (1).xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Inspect the columns to ensure they match expectations
print(df.columns)

# Assuming columns match amino acids (e.g. 'HIS' for Histidine, etc.), map to reference names
amino_acid_mapping = {
    'HIS': 'Histidine',
    'ILE': 'Isoleucine',
    'LEU': 'Leucine',
    'LYS': 'Lysine',
    'MET+CYS': 'Methionine + Cysteine',
    'PHE+TYR': 'Phenylalanine + Tyrosine',
    'THR': 'Threonine',
    'TRP': 'Tryptophan',
    'VAL': 'Valine'
}

# Rename the columns to standard names (ensure these match your data)
df.rename(columns=amino_acid_mapping, inplace=True)

# Fill missing values (if any) with 0 to avoid errors in calculations
df.fillna(0, inplace=True)

# Function to calculate amino acid score
def calculate_amino_acid_score(protein_row, reference_pattern):
    scores = {}
    for aa in reference_pattern.keys():
        if aa in protein_row:
            # Amino Acid Score = (mg of amino acid in test protein) / (mg of amino acid in reference pattern)
            scores[aa] = protein_row[aa] / reference_pattern[aa] if reference_pattern[aa] != 0 else 0
    return scores

# Calculate scores for each reference pattern
score_results = {age_group: [] for age_group in reference_patterns.keys()}

for age_group, reference_pattern in reference_patterns.items():
    for index, row in df.iterrows():
        sample_name = row['SAMPLE']  # Get the protein source name
        amino_acid_scores = calculate_amino_acid_score(row, reference_pattern)
       
        # Find the limiting amino acid and its score
        limiting_aa = min(amino_acid_scores, key=amino_acid_scores.get)
        limiting_score = amino_acid_scores[limiting_aa]
       
        result = {
            'SAMPLE': sample_name,
            'Limiting Amino Acid': limiting_aa,
            'Amino Acid Score': limiting_score
        }
        score_results[age_group].append(result)

# Convert the results into DataFrames and save them
for age_group, results in score_results.items():
    score_df = pd.DataFrame(results)
    output_file = fr'C:\Users\hp\Desktop\project\{age_group}_Amino_Acid_Score_Results.xlsx'
    score_df.to_excel(output_file, index=False)

print("Results have been saved for all age groups.")
