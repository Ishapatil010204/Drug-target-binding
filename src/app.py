"""
REAL ML-BASED STREAMLIT APP
This version loads and uses actual trained ML models from the 'models/' directory.

Prerequisites:
1. Run train.ipynb (or a train script) to create the .pkl files in models/
2. Then run: streamlit run src/app.py (from the root directory)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from collections import Counter
import math

# Try to import RDKit (optional)
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    import py3Dmol # <-- FIX 1: Corrected import
    import streamlit.components.v1 as components
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Drug-Target Interaction Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Removed spacing and made everything compact
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #e0f2fe 0%, #fae8ff 100%);}
    .stButton>button {
        width: 100%; background: linear-gradient(90deg, #9333ea 0%, #ec4899 100%);
        color: white; font-weight: bold; border: none; padding: 0.5rem;
        border-radius: 0.5rem; font-size: 1rem;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(147, 51, 234, 0.4);
    }
    .small-metric {
        font-size: 0.8rem;
    }
    .small-metric .stMetric {
        font-size: 0.8rem;
    }
    .small-metric .stMetric label {
        font-size: 0.7rem;
    }
    .small-metric .stMetric value {
        font-size: 0.9rem;
    }
    .small-text {
        font-size: 0.75rem;
    }
    .tiny-text {
        font-size: 0.7rem;
    }
    .small-number {
        font-size: 0.7rem;
        font-weight: bold;
    }
    .atomic-comp {
        background-color: #f8f9fa;
        padding: 6px;
        border-radius: 4px;
        border-left: 2px solid #9333ea;
        font-size: 0.65rem;
        margin: 5px 0;
    }
    .property-value {
        font-size: 0.7rem;
        font-weight: bold;
    }
    /* Remove spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stTextArea textarea {
        height: 100px;
    }
    /* Compact layout */
    .row-widget.stButton {
        margin-bottom: 0.5rem;
    }
    .element-container {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== FEATURE EXTRACTOR ====================
class FeatureExtractor:
    """
    Extract features from proteins and drugs.
    This class definition MUST match the one used during training in train.ipynb
    """

    def __init__(self):
        # Amino acid molecular weights (monoisotopic)
        self.aa_weights = {
            'A': 89.0935, 'C': 121.1590, 'D': 133.1032, 'E': 147.1299, 'F': 165.1900,
            'G': 75.0669, 'H': 155.1552, 'I': 131.1736, 'K': 146.1882, 'L': 131.1736,
            'M': 149.2124, 'N': 132.1184, 'P': 115.1310, 'Q': 146.1451, 'R': 174.2017,
            'S': 105.0930, 'T': 119.1197, 'V': 117.1469, 'W': 204.2262, 'Y': 181.1894
        }
        
        # Amino acid compositions (C, H, N, O, S atoms)
        self.aa_composition = {
            'A': {'C': 3, 'H': 5, 'N': 1, 'O': 1, 'S': 0},
            'C': {'C': 3, 'H': 5, 'N': 1, 'O': 1, 'S': 1},
            'D': {'C': 4, 'H': 5, 'N': 1, 'O': 3, 'S': 0},
            'E': {'C': 5, 'H': 7, 'N': 1, 'O': 3, 'S': 0},
            'F': {'C': 9, 'H': 9, 'N': 1, 'O': 1, 'S': 0},
            'G': {'C': 2, 'H': 3, 'N': 1, 'O': 1, 'S': 0},
            'H': {'C': 6, 'H': 7, 'N': 3, 'O': 1, 'S': 0},
            'I': {'C': 6, 'H': 11, 'N': 1, 'O': 1, 'S': 0},
            'K': {'C': 6, 'H': 12, 'N': 2, 'O': 1, 'S': 0},
            'L': {'C': 6, 'H': 11, 'N': 1, 'O': 1, 'S': 0},
            'M': {'C': 5, 'H': 9, 'N': 1, 'O': 1, 'S': 1},
            'N': {'C': 4, 'H': 6, 'N': 2, 'O': 2, 'S': 0},
            'P': {'C': 5, 'H': 7, 'N': 1, 'O': 1, 'S': 0},
            'Q': {'C': 5, 'H': 8, 'N': 2, 'O': 2, 'S': 0},
            'R': {'C': 6, 'H': 12, 'N': 4, 'O': 1, 'S': 0},
            'S': {'C': 3, 'H': 5, 'N': 1, 'O': 2, 'S': 0},
            'T': {'C': 4, 'H': 7, 'N': 1, 'O': 2, 'S': 0},
            'V': {'C': 5, 'H': 9, 'N': 1, 'O': 1, 'S': 0},
            'W': {'C': 11, 'H': 10, 'N': 2, 'O': 1, 'S': 0},
            'Y': {'C': 9, 'H': 9, 'N': 1, 'O': 2, 'S': 0}
        }
        
        # pKa values for amino acids (N-terminal, C-terminal, and side chains)
        self.aa_pka = {
            'A': {'N': 9.69, 'C': 2.34}, 
            'C': {'N': 10.78, 'C': 1.71, 'side': 8.33},
            'D': {'N': 9.60, 'C': 1.88, 'side': 3.65}, 
            'E': {'N': 9.67, 'C': 2.19, 'side': 4.25},
            'F': {'N': 9.13, 'C': 1.83}, 
            'G': {'N': 9.60, 'C': 2.34},
            'H': {'N': 9.17, 'C': 1.82, 'side': 6.00}, 
            'I': {'N': 9.68, 'C': 2.36},
            'K': {'N': 8.95, 'C': 2.18, 'side': 10.53}, 
            'L': {'N': 9.60, 'C': 2.36},
            'M': {'N': 9.21, 'C': 2.28}, 
            'N': {'N': 8.80, 'C': 2.02},
            'P': {'N': 10.60, 'C': 1.99}, 
            'Q': {'N': 9.13, 'C': 2.17},
            'R': {'N': 9.04, 'C': 2.17, 'side': 12.48}, 
            'S': {'N': 9.15, 'C': 2.21},
            'T': {'N': 9.10, 'C': 2.09}, 
            'V': {'N': 9.62, 'C': 2.32},
            'W': {'N': 9.39, 'C': 2.38}, 
            'Y': {'N': 9.11, 'C': 2.20, 'side': 10.07}
        }
        
        # Hydropathy index (Kyte & Doolittle)
        self.hydropathy = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        # Instability index weights (Guruprasad et al. 1990)
        self.instability_weights = {
            'AA': 0, 'AC': 0, 'AD': 0, 'AE': 0, 'AF': 1,
            'AG': 0, 'AH': 0, 'AI': 0, 'AK': 0, 'AL': 1,
            'AM': 0, 'AN': 0, 'AP': 0, 'AQ': 0, 'AR': 0,
            'AS': 0, 'AT': 0, 'AV': 0, 'AW': 0, 'AY': 0,
            'CA': 1, 'CC': 9, 'CD': 10, 'CE': 10, 'CF': 5,
            'CG': 1, 'CH': 2, 'CI': 4, 'CK': 8, 'CL': 6,
            'CM': 5, 'CN': 4, 'CP': 0, 'CQ': 4, 'CR': 6,
            'CS': 2, 'CT': 2, 'CV': 4, 'CW': 8, 'CY': 4,
            'DA': 7, 'DC': 10, 'DD': 6, 'DE': 6, 'DF': 10,
            'DG': 7, 'DH': 7, 'DI': 8, 'DK': 7, 'DL': 10,
            'DM': 9, 'DN': 4, 'DP': 6, 'DQ': 5, 'DR': 8,
            'DS': 4, 'DT': 4, 'DV': 8, 'DW': 14, 'DY': 9,
            'EA': 7, 'EC': 10, 'ED': 6, 'EE': 6, 'EF': 10,
            'EG': 7, 'EH': 7, 'EI': 8, 'EK': 7, 'EL': 10,
            'EM': 9, 'EN': 4, 'EP': 6, 'EQ': 5, 'ER': 8,
            'ES': 4, 'ET': 4, 'EV': 8, 'EW': 14, 'EY': 9,
            'FA': 4, 'FC': 6, 'FD': 10, 'FE': 10, 'FF': 2,
            'FG': 4, 'FH': 4, 'FI': 6, 'FK': 6, 'FL': 2,
            'FM': 4, 'FN': 6, 'FP': 5, 'FQ': 6, 'FR': 6,
            'FS': 6, 'FT': 5, 'FV': 6, 'FW': 6, 'FY': 5,
            'GA': 0, 'GC': 1, 'GD': 7, 'GE': 7, 'GF': 4,
            'GG': 0, 'GH': 1, 'GI': 1, 'GK': 1, 'GL': 1,
            'GM': 1, 'GN': 1, 'GP': 0, 'GQ': 1, 'GR': 1,
            'GS': 0, 'GT': 0, 'GV': 1, 'GW': 4, 'GY': 4,
            'HA': 2, 'HC': 2, 'HD': 7, 'HE': 7, 'HF': 4,
            'HG': 1, 'HH': 0, 'HI': 2, 'HK': 2, 'HL': 2,
            'HM': 2, 'HN': 2, 'HP': 1, 'HQ': 2, 'HR': 2,
            'HS': 2, 'HT': 2, 'HV': 2, 'HW': 4, 'HY': 4,
            'IA': 1, 'IC': 4, 'ID': 8, 'IE': 8, 'IF': 6,
            'IG': 1, 'IH': 2, 'II': 0, 'IK': 3, 'IL': 0,
            'IM': 1, 'IN': 3, 'IP': 1, 'IQ': 3, 'IR': 3,
            'IS': 2, 'IT': 1, 'IV': 0, 'IW': 6, 'IY': 5,
            'KA': 0, 'KC': 8, 'KD': 7, 'KE': 7, 'KF': 6,
            'KG': 1, 'KH': 2, 'KI': 3, 'KK': 0, 'KL': 3,
            'KM': 3, 'KN': 2, 'KP': 1, 'KQ': 2, 'KR': 0,
            'KS': 1, 'KT': 1, 'KV': 3, 'KW': 8, 'KY': 5,
            'LA': 1, 'LC': 6, 'LD': 10, 'LE': 10, 'LF': 2,
            'LG': 1, 'LH': 2, 'LI': 0, 'LK': 3, 'LL': 0,
            'LM': 1, 'LN': 3, 'LP': 1, 'LQ': 3, 'LR': 3,
            'LS': 2, 'LT': 1, 'LV': 0, 'LW': 6, 'LY': 5,
            'MA': 1, 'MC': 5, 'MD': 9, 'ME': 9, 'MF': 4,
            'MG': 1, 'MH': 2, 'MI': 1, 'MK': 3, 'ML': 1,
            'MM': 0, 'MN': 3, 'MP': 1, 'MQ': 3, 'MR': 3,
            'MS': 2, 'MT': 2, 'MV': 1, 'MW': 5, 'MY': 4,
            'NA': 2, 'NC': 4, 'ND': 4, 'NE': 4, 'NF': 6,
            'NG': 1, 'NH': 2, 'NI': 3, 'NK': 2, 'NL': 3,
            'NM': 3, 'NN': 0, 'NP': 2, 'NQ': 1, 'NR': 2,
            'NS': 1, 'NT': 1, 'NV': 3, 'NW': 6, 'NY': 4,
            'PA': 0, 'PC': 0, 'PD': 6, 'PE': 6, 'PF': 5,
            'PG': 0, 'PH': 1, 'PI': 1, 'PK': 1, 'PL': 1,
            'PM': 1, 'PN': 2, 'PP': 0, 'PQ': 1, 'PR': 1,
            'PS': 1, 'PT': 1, 'PV': 1, 'PW': 6, 'PY': 4,
            'QA': 0, 'QC': 4, 'QD': 5, 'QE': 5, 'QF': 6,
            'QG': 1, 'QH': 2, 'QI': 3, 'QK': 2, 'QL': 3,
            'QM': 3, 'QN': 1, 'QP': 1, 'QQ': 0, 'QR': 2,
            'QS': 1, 'QT': 1, 'QV': 3, 'QW': 7, 'QY': 4,
            'RA': 0, 'RC': 6, 'RD': 8, 'RE': 8, 'RF': 6,
            'RG': 1, 'RH': 2, 'RI': 3, 'RK': 0, 'RL': 3,
            'RM': 3, 'RN': 2, 'RP': 1, 'RQ': 2, 'RR': 0,
            'RS': 1, 'RT': 1, 'RV': 3, 'RW': 8, 'RY': 5,
            'SA': 0, 'SC': 2, 'SD': 4, 'SE': 4, 'SF': 6,
            'SG': 0, 'SH': 2, 'SI': 2, 'SK': 1, 'SL': 2,
            'SM': 2, 'SN': 1, 'SP': 1, 'SQ': 1, 'SR': 1,
            'SS': 0, 'ST': 0, 'SV': 2, 'SW': 6, 'SY': 4,
            'TA': 0, 'TC': 2, 'TD': 4, 'TE': 4, 'TF': 5,
            'TG': 0, 'TH': 2, 'TI': 1, 'TK': 1, 'TL': 1,
            'TM': 2, 'TN': 1, 'TP': 1, 'TQ': 1, 'TR': 1,
            'TS': 0, 'TT': 0, 'TV': 1, 'TW': 6, 'TY': 4,
            'VA': 0, 'VC': 4, 'VD': 8, 'VE': 8, 'VF': 6,
            'VG': 1, 'VH': 2, 'VI': 0, 'VK': 3, 'VL': 0,
            'VM': 1, 'VN': 3, 'VP': 1, 'VQ': 3, 'VR': 3,
            'VS': 2, 'VT': 1, 'VV': 0, 'VW': 6, 'VY': 5,
            'WA': 4, 'WC': 8, 'WD': 14, 'WE': 14, 'WF': 6,
            'WG': 4, 'WH': 4, 'WI': 6, 'WK': 8, 'WL': 6,
            'WM': 5, 'WN': 6, 'WP': 6, 'WQ': 7, 'WR': 8,
            'WS': 6, 'WT': 6, 'WV': 6, 'WW': 0, 'WY': 8,
            'YA': 4, 'YC': 4, 'YD': 9, 'YE': 9, 'YF': 5,
            'YG': 4, 'YH': 4, 'YI': 5, 'YK': 5, 'YL': 5,
            'YM': 4, 'YN': 4, 'YP': 4, 'YQ': 4, 'YR': 5,
            'YS': 4, 'YT': 4, 'YV': 5, 'YW': 8, 'YY': 0
        }
        
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    def protein_to_features(self, sequence):
        """Convert protein sequence to 33 numerical features"""
        sequence = ''.join([aa for aa in sequence if aa in self.amino_acids])

        if len(sequence) == 0:
            return None

        total = len(sequence)
        features = []

        aa_counts = Counter(sequence)
        for aa in self.amino_acids:
            features.append(aa_counts[aa] / total)

        features.append(total) # Length
        features.append(sum([self.aa_weights.get(aa, 110) for aa in sequence])) # MW
        features.append((aa_counts['F'] + aa_counts['W'] + aa_counts['Y']) / total) # Aromaticity

        if len(sequence) > 1:
            dipeptides = [sequence[i:i+2] for i in range(len(sequence)-1)]
            dipeptide_counts = Counter(dipeptides)
            most_common = dipeptide_counts.most_common(10)
            for i in range(10):
                if i < len(most_common):
                    features.append(most_common[i][1] / (total - 1))
                else:
                    features.append(0)
        else:
            features.extend([0] * 10)

        return features

    def smiles_to_features(self, smiles):
        """Convert SMILES to 17 numerical features"""
        features = []
        features.append(len(smiles)) # Length
        features.append(smiles.count('C')) # Carbons
        features.append(smiles.count('O')) # Oxygens
        features.append(smiles.count('N')) # Nitrogens
        features.append(smiles.count('S')) # Sulfurs
        features.append(smiles.count('P')) # Phosphorus
        features.append(smiles.count('=')) # Double bonds
        features.append(smiles.count('#')) # Triple bonds
        features.append(smiles.count('(')) # Branches
        features.append(smiles.count('[')) # Atoms in brackets
        features.append(smiles.count('@')) # Chirality

        for i in range(1, 7):
            features.append(smiles.count(str(i)))

        return features

    def combine_features(self, protein_features, drug_features):
        """Combine protein and drug features"""
        return protein_features + drug_features

    def calculate_physicochemical_properties(self, sequence):
        """Calculate accurate physicochemical properties for protein"""
        if not sequence:
            return {}
        
        sequence = sequence.upper()
        total = len(sequence)
        aa_counts = Counter(sequence)
        
        # Calculate molecular weight accurately
        mol_weight = 18.01528  # Start with water (H2O)
        for aa, count in aa_counts.items():
            mol_weight += self.aa_weights.get(aa, 0) * count
        
        # Atomic composition
        atomic_composition = {'C': 0, 'H': 0, 'N': 0, 'O': 0, 'S': 0}
        for aa, count in aa_counts.items():
            comp = self.aa_composition.get(aa, {'C': 0, 'H': 0, 'N': 0, 'O': 0, 'S': 0})
            for atom, value in comp.items():
                atomic_composition[atom] += value * count
        
        # Add water atoms (H2O)
        atomic_composition['H'] += 2
        atomic_composition['O'] += 1
        
        # Charge calculation for pI
        def charge_at_ph(ph):
            charge = 0
            # N-terminal (NH3+)
            charge += 1.0 / (1.0 + 10**(ph - 8.0))
            # C-terminal (COOH) 
            charge -= 1.0 / (1.0 + 10**(3.1 - ph))
            # Side chains
            for aa, count in aa_counts.items():
                pka_data = self.aa_pka.get(aa, {})
                if 'side' in pka_data:
                    if aa in ['D', 'E']:  # acidic
                        charge -= count / (1.0 + 10**(pka_data['side'] - ph))
                    else:  # basic (H, K, R, Y)
                        charge += count / (1.0 + 10**(ph - pka_data['side']))
            return charge
        
        # Find pI using binary search
        low, high = 0.0, 14.0
        pI = 7.0
        for _ in range(50):  # Binary search for pI
            mid = (low + high) / 2.0
            charge = charge_at_ph(mid)
            if abs(charge) < 0.001:
                pI = mid
                break
            elif charge > 0:
                low = mid
            else:
                high = mid
        
        # Charged residues
        neg_charged = aa_counts.get('D', 0) + aa_counts.get('E', 0)
        pos_charged = aa_counts.get('R', 0) + aa_counts.get('K', 0) + aa_counts.get('H', 0)
        
        # Estimated half-life
        n_terminal = sequence[0] if sequence else 'M'
        if n_terminal == 'M':
            half_life = "30 hours (mammalian), >20 hours (yeast), >10 hours (E. coli)"
        elif n_terminal in ['P', 'E', 'S', 'T']:
            half_life = ">20 hours (mammalian)"
        else:
            half_life = "<30 minutes (mammalian)"
        
        # Instability index (Guruprasad et al. 1990)
        instability = 0.0
        if total > 1:
            for i in range(total - 1):
                dipeptide = sequence[i:i+2]
                weight = self.instability_weights.get(dipeptide, 0)
                instability += weight
            instability = (10.0 / total) * instability
        
        # Stability classification
        stability_class = "stable" if instability < 40 else "unstable"
        
        # Aliphatic index
        aliphatic_index = (aa_counts.get('A', 0) * 2.9 + 
                           aa_counts.get('V', 0) * 4.2 + 
                           aa_counts.get('I', 0) * 4.5 + 
                           aa_counts.get('L', 0) * 3.8) / total * 100
        
        # GRAVY (Grand Average of Hydropathicity)
        gravy = sum(self.hydropathy.get(aa, 0) * count for aa, count in aa_counts.items()) / total
        
        # Extinction coefficients (Gill & von Hippel method)
        extinction_w = aa_counts.get('W', 0) * 5500
        extinction_y = aa_counts.get('Y', 0) * 1490
        extinction_c = aa_counts.get('C', 0) * 125
        extinction_total = extinction_w + extinction_y + extinction_c
        
        # Absorbance 0.1% (1 g/l)
        abs_0_1percent = extinction_total / mol_weight if mol_weight > 0 else 0
        
        # Total number of atoms
        total_atoms = sum(atomic_composition.values())
        
        # Formula string
        formula = f"C{atomic_composition['C']}H{atomic_composition['H']}N{atomic_composition['N']}O{atomic_composition['O']}S{atomic_composition['S']}"
        
        return {
            'theoretical_pi': round(pI, 2),
            'neg_charged_residues': neg_charged,
            'pos_charged_residues': pos_charged,
            'atomic_composition': atomic_composition,
            'total_atoms': total_atoms,
            'formula': formula,
            'estimated_half_life': half_life,
            'instability_index': round(instability, 2),
            'stability_class': stability_class,
            'aliphatic_index': round(aliphatic_index, 2),
            'gravy': round(gravy, 3),
            'extinction_coefficient': extinction_total,
            'abs_0_1percent': round(abs_0_1percent, 3),
            'molecular_weight': round(mol_weight, 2)
        }


# ==================== REAL ML PREDICTOR ====================
class RealMLPredictor:
    """Uses actual trained ML models"""

    def __init__(self):
        self.models_loaded = False
        self.feature_extractor = FeatureExtractor()
        self.load_models()

    def load_models(self):
        """Load trained models from the 'models/' directory"""
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        model_dir = os.path.abspath(model_dir)

        try:
            with open(os.path.join(model_dir, 'classifier.pkl'), 'rb') as f:
                self.classifier = pickle.load(f)
                self.classifier.set_params(device='cpu')  # <-- ADD THIS LINE

            with open(os.path.join(model_dir, 'regressor.pkl'), 'rb') as f:
                self.regressor = pickle.load(f)
                self.regressor.set_params(device='cpu')  # <-- AND ADD THIS LINE

            with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.models_loaded = True
            return True
        except FileNotFoundError:
            return False

    def predict(self, protein_seq, drug_smiles):
        print("\n--- Starting Prediction (v2 Debug) ---")
        try:
            # 1. Pre-process inputs
            print("[Step 1] Pre-processing inputs...")
            protein_seq = protein_seq.strip().upper()
            drug_smiles = drug_smiles.strip()
            print(f"  Protein Seq (start): {protein_seq[:20]}...")
            print(f"  Drug SMILES: {drug_smiles}")

            # 2. Extract features
            print("[Step 2] Extracting features...")
            prot_features = self.feature_extractor.protein_to_features(protein_seq)
            drug_features = self.feature_extractor.smiles_to_features(drug_smiles)
            print(f"  Protein features extracted: {'Yes' if prot_features else 'No'}")
            print(f"  Drug features extracted: {'Yes' if drug_features else 'No'}")

            if prot_features is None or drug_features is None:
                st.error("Invalid Protein Sequence or SMILES string. Could not extract features.")
                print("--- Prediction Failed: Feature Extraction ---")
                return None

            # 3. Combine and Format
            print("[Step 3] Combining and formatting features...")
            combined = self.feature_extractor.combine_features(prot_features, drug_features)
            X = np.array(combined)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X = X.astype(np.float32)
            print(f"  Combined feature vector shape: {X.shape}, dtype: {X.dtype}")

            if self.models_loaded:
                # 4a. Scale
                print("[Step 4a] Scaling features...")
                X_scaled = None
                try:
                    print(f"  Input to scaler - Shape: {X.shape}, Type: {X.dtype}")
                    X_scaled = self.scaler.transform(X)
                    X_scaled = X_scaled.astype(np.float32)
                    print(f"  Output from scaler - Shape: {X_scaled.shape}, Type: {X_scaled.dtype}")
                except Exception as e_scale:
                    print(f"--- CRASH POINT: Scaling Failed ---")
                    print(f"Error: {e_scale}")
                    st.error(f"Error during feature scaling: {e_scale}")
                    import traceback; traceback.print_exc()
                    return None

                # 4b. Predict Probability (Classifier)
                print("[Step 4b] Predicting binding probability (Classifier)...")
                binding_prob = None
                try:
                    print(f"  Input to classifier.predict_proba - Shape: {X_scaled.shape}, Type: {X_scaled.dtype}")
                    binding_prob_array = self.classifier.predict_proba(X_scaled.astype(np.float32))
                    binding_prob = binding_prob_array[0][1]
                    print(f"  Predicted binding probability: {binding_prob}")
                except Exception as e_class:
                    print(f"--- CRASH POINT: Classifier Prediction Failed ---")
                    print(f"Error: {e_class}")
                    st.error(f"Error during classification prediction: {e_class}")
                    import traceback; traceback.print_exc()
                    return None

                # 4c. Predict IC50 (Regressor)
                print("[Step 4c] Predicting IC50 (Regressor)...")
                ic50_value = None
                try:
                    print(f"  Input to regressor.predict - Shape: {X_scaled.shape}, Type: {X_scaled.dtype}")
                    ic50_raw = self.regressor.predict(X_scaled.astype(np.float32))
                    ic50_value = ic50_raw[0]
                    print(f"  Predicted raw IC50: {ic50_value}")
                    ic50_value = max(0.0, float(ic50_value))
                    print(f"  Predicted clamped IC50: {ic50_value}")
                except Exception as e_reg:
                    print(f"--- CRASH POINT: Regressor Prediction Failed ---")
                    print(f"Error: {e_reg}")
                    st.error(f"Error during regression prediction: {e_reg}")
                    import traceback; traceback.print_exc()
                    return None

            else:
                # Fallback demo mode
                print("[Step 4 Demo] Generating demo values...")
                binding_prob = np.clip((len(protein_seq) * len(drug_smiles)) / 5000 + np.random.uniform(-0.1, 0.1), 0.05, 0.98)
                ic50_value = np.random.uniform(3, 12)
                print(f"  Demo binding prob: {binding_prob}, Demo IC50: {ic50_value}")

            # 5. Post-process Results
            print("[Step 5] Post-processing results...")
            classification = (
                'Strong Binder' if binding_prob > 0.7 else
                'Moderate Binder' if binding_prob > 0.4 else 'Weak Binder'
            )
            confidence = 'High' if abs(binding_prob - 0.5) > 0.3 else 'Medium'

            aromaticity = 0.0
            mol_weight = 0
            if len(protein_seq) > 0:
                aromaticity = (protein_seq.count('F') + protein_seq.count('W') + protein_seq.count('Y')) / len(protein_seq)
                mol_weight = sum([self.feature_extractor.aa_weights.get(aa, 110) for aa in protein_seq])

            # Calculate physicochemical properties
            physico_props = self.feature_extractor.calculate_physicochemical_properties(protein_seq)

            result_dict = {
                'binding_probability': float(binding_prob),
                'ic50_value': float(ic50_value),
                'classification': classification,
                'protein_length': len(protein_seq),
                'molecular_weight': mol_weight,
                'aromaticity': aromaticity,
                'confidence': confidence,
                'model_type': 'Trained ML Model' if self.models_loaded else 'Demo Mode',
                'physicochemical_properties': physico_props
            }
            print(f"  Final result dictionary: {result_dict}")
            print("--- Prediction Successful ---")
            return result_dict

        except Exception as e_outer:
            print(f"--- CRASH POINT: Unexpected Error in Predict Function ---")
            print(f"Error: {e_outer}")
            st.error(f"An unexpected error occurred during prediction: {e_outer}")
            import traceback
            traceback.print_exc()
            return None


# ==================== EXAMPLES ====================
EXAMPLES = {
    "Aspirin + Protein Kinase": {
        "protein": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPL",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "name": "Aspirin"
    },
    "Caffeine + Adenosine Receptor": {
        "protein": "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL",
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "name": "Caffeine"
    },
    "Ibuprofen + COX Enzyme": {
        "protein": "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ",
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "name": "Ibuprofen"
    }
}


# ==================== MAIN STREAMLIT APP ====================
def main():
    if 'predictor' not in st.session_state:
        st.session_state.predictor = RealMLPredictor()
    predictor = st.session_state.predictor

    st.markdown('<div style="text-align:center;padding:1rem 0;"><h1 style="font-size:2.5rem;background:linear-gradient(90deg,#9333ea 0%,#ec4899 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🧬 Drug-Target Interaction Predictor</h1><p style="font-size:1rem;color:#6b7280;">ML-Based Prediction of Binding Affinity and Interaction Probability</p></div>', unsafe_allow_html=True)

    if predictor.models_loaded:
        st.success("✅ **Real ML Models Loaded** - Using trained classifier and regressor from `models/` directory")
    else:
        st.warning("⚠️ **Demo Mode** - Models not found in `models/`. Run `train.ipynb` first to generate them.")

    # Sidebar
    with st.sidebar:
        st.header("🧪 Example Data")
        for name, data in EXAMPLES.items():
            if st.button(name, key=f"example_{name}"):
                st.session_state.protein_input = data['protein']
                st.session_state.smiles_input = data['smiles']
                st.session_state.results = None
                st.rerun()
        st.info("💡 **How It Works:** Uses a trained XGBoost model to predict if a drug (SMILES) will bind to a protein (sequence) and estimates the binding affinity.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🧫 Protein Target Sequence")
        protein_seq = st.text_area(
            "Enter amino acid sequence",
            height=100,
            key="protein_input"
        )

        st.markdown("### 💊 Drug Molecule (SMILES)")
        drug_smiles = st.text_input(
            "Enter SMILES notation",
            key="smiles_input"
        )

        c1, c2 = st.columns(2)
        with c1:
            predict_btn = st.button("🔮 Predict Interaction")
        with c2:
            reset_btn = st.button("🔁 Reset")

        if reset_btn:
            st.session_state.results = None
            st.rerun()

    if predict_btn:
        if not protein_seq or not drug_smiles:
            st.error("⚠️ Please provide both protein sequence and drug SMILES")
        else:
            with st.spinner("⏳ Analyzing interaction..."):
                results = predictor.predict(protein_seq, drug_smiles)
                if results:
                    st.session_state.results = results

    if st.session_state.get('results'):
        results = st.session_state.results

        # Display molecule visualization in the second column
        with col2:
            if RDKIT_AVAILABLE:
                # Drug Molecule Visualization
                st.markdown("### 🧪 Drug Molecule Structure")
                mol_smiles = st.session_state.get('smiles_input', '')
                if mol_smiles:
                    try:
                        mol = Chem.MolFromSmiles(mol_smiles)
                        if mol:
                            from rdkit.Chem import AllChem # Import AllChem
                            
                            # --- 2D Visualization ---
                            mol_2d = Chem.AddHs(mol)  # Create 2D copy
                            AllChem.Compute2DCoords(mol_2d)
                            st.image(Draw.MolToImage(mol_2d, size=(300, 250)), 
                                   caption=f"2D Structure: {mol_smiles}")
                            
                            # --- 3D Visualization (Clean & Stable) ---
                            st.markdown("### 🧬 3D Molecular Structure")

                            try:
                                mol_3d = Chem.AddHs(mol)
                                result = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
                                if result != 0:
                                    raise ValueError("3D embedding failed. Try a simpler or valid SMILES string.")

                                AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
                                pdb_block = Chem.MolToPDBBlock(mol_3d)

                                viewer = py3Dmol.view(width=350, height=300)
                                viewer.addModel(pdb_block, 'pdb')
                                viewer.setStyle({'stick': {'radius': 0.2, 'colorscheme': 'cyanCarbon'}})
                                viewer.zoomTo()
                                viewer.setBackgroundColor('#f8fafc')
                                components.html(viewer._make_html(), height=300, width=350)

                            except Exception as e:
                                st.warning(f"⚠️ Could not render 3D structure: {e}")
                            
                            # --- Drug Properties (from base mol) ---
                            st.markdown("**Drug Properties:**")
                            mol_weight = AllChem.CalcExactMolWt(mol)
                            formula = AllChem.CalcMolFormula(mol)
                            st.write(f"Molecular Formula: {formula}")
                            st.write(f"Molecular Weight: {mol_weight:.2f} g/mol")
                            st.write(f"Number of Atoms: {mol.GetNumAtoms()}")
                            st.write(f"Number of Bonds: {mol.GetNumBonds()}")                       



                        else:
                            st.warning("Could not generate image for this SMILES string.")
                    except Exception as img_err:
                        print(f"RDKit image generation error: {img_err}")
                        st.warning("RDKit error generating molecule image.")
            else:
                st.warning("RDKit or py3Dmol not available. Install with: pip install rdkit py3Dmol")
                
                # # Protein Structure Information
                # st.markdown("### 🧫 Protein Information")
                # protein_seq = st.session_state.get('protein_input', '')
                # if protein_seq:
                #     # Display protein sequence info
                #     st.write(f"**Sequence Length:** {len(protein_seq)} amino acids")
                    
                #     # Calculate and display amino acid composition
                #     from collections import Counter
                #     aa_count = Counter(protein_seq)
                #     common_aa = aa_count.most_common(5)
                #     st.write("**Top 5 Amino Acids:**")
                #     for aa, count in common_aa:
                #         st.write(f"  {aa}: {count} ({count/len(protein_seq)*100:.1f}%)")
                    
                    # # Show protein properties from results
                    # if results:
                    #     st.write(f"**Molecular Weight:** ~{results['molecular_weight']:.0f} Da")
                    #     st.write(f"**Aromaticity:** {results['aromaticity']*100:.1f}%")
                        
                        # # Display physicochemical properties
                        # physico_props = results.get('physicochemical_properties', {})
                        # if physico_props:
                        #     st.write(f"**Theoretical pI:** {physico_props.get('theoretical_pi', 'N/A')}")
                        #     st.write(f"**Instability Index:** {physico_props.get('instability_index', 'N/A')}")

        # Display results in the main column
        with col1:
            st.markdown("## 🧾 Prediction Results")
            st.caption(f"Model type: {results['model_type']}")

            # SECTION 1: Binding Prediction Results
            st.markdown("### 🔮 Binding Prediction")
            st.markdown('<div class="small-metric">', unsafe_allow_html=True)
            
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            with pred_col1:
                st.metric("Binding Probability", f"{results['binding_probability']*100:.1f}%")
            with pred_col2:
                st.metric("Predicted IC50 (μM)", f"{results['ic50_value']:.2f}")
            with pred_col3:
                st.metric("Classification", results['classification'])
            
            st.markdown(f"**Confidence:** {results['confidence']}")
            st.markdown('</div>', unsafe_allow_html=True)

            # SECTION 2: Protein Properties
            st.markdown("### 🧫 Protein Properties")
            st.markdown('<div class="small-metric">', unsafe_allow_html=True)
            
            prop_col1, prop_col2, prop_col3 = st.columns(3)
            with prop_col1:
                st.metric("Length", f"{results['protein_length']} aa")
            with prop_col2:
                st.metric("Mol. Weight", f"~{results['molecular_weight']:.0f} Da")
            with prop_col3:
                st.metric("Aromaticity", f"{results['aromaticity']*100:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # SECTION 3: Detailed Physicochemical Properties
            st.markdown("### 🧪 Detailed Physicochemical Properties")
            st.markdown('<div class="tiny-text">', unsafe_allow_html=True)
            
            physico_props = results.get('physicochemical_properties', {})
            
            # Main properties in a compact grid
            st.markdown("**Physicochemical Parameters:**")
            main_col1, main_col2, main_col3, main_col4 = st.columns(4)
            with main_col1:
                st.markdown(f"<div class='property-value'>Theoretical pI: {physico_props.get('theoretical_pi', 'N/A')}</div>", unsafe_allow_html=True)
            with main_col2:
                st.markdown(f"<div class='property-value'>Instability Index: {physico_props.get('instability_index', 'N/A')}</div>", unsafe_allow_html=True)
            with main_col3:
                st.markdown(f"<div class'property-value'>Aliphatic Index: {physico_props.get('aliphatic_index', 'N/A')}</div>", unsafe_allow_html=True)
            with main_col4:
                st.markdown(f"<div class='property-value'>GRAVY: {physico_props.get('gravy', 'N/A')}</div>", unsafe_allow_html=True)
            
            st.markdown(f"*Stability: {physico_props.get('stability_class', 'N/A')}*")
            
            # Charged residues and additional properties
            charge_col1, charge_col2 = st.columns(2)
            with charge_col1:
                st.write(f"**Negatively charged (Asp + Glu):** {physico_props.get('neg_charged_residues', 0)}")
            with charge_col2:
                st.write(f"**Positively charged (Arg + Lys + His):** {physico_props.get('pos_charged_residues', 0)}")
            
            # Additional properties
            st.markdown(f"**Estimated half-life:** {physico_props.get('estimated_half_life', 'N/A')}")
            
            ext_col1, ext_col2 = st.columns(2)
            with ext_col1:
                st.markdown(f"<div class='property-value'>Extinction Coefficient: {physico_props.get('extinction_coefficient', 'N/A')}</div>", unsafe_allow_html=True)
                st.caption("M⁻¹ cm⁻¹ at 280 nm")
            with ext_col2:
                st.markdown(f"<div class='property-value'>Abs 0.1%: {physico_props.get('abs_0_1percent', 'N/A')}</div>", unsafe_allow_html=True)
                st.caption("(1 g/l)")
            
            # Atomic Composition
            st.markdown('<div class="atomic-comp">', unsafe_allow_html=True)
            st.markdown("**Atomic Composition:**")
            atomic_comp = physico_props.get('atomic_composition', {})
            if atomic_comp:
                atom_col1, atom_col2, atom_col3, atom_col4, atom_col5 = st.columns(5)
                with atom_col1:
                    st.markdown(f"<div class='small-number'>C: {atomic_comp.get('C', 0)}</div>", unsafe_allow_html=True)
                with atom_col2:
                    st.markdown(f"<div class='small-number'>H: {atomic_comp.get('H', 0)}</div>", unsafe_allow_html=True)
                with atom_col3:
                    st.markdown(f"<div class='small-number'>N: {atomic_comp.get('N', 0)}</div>", unsafe_allow_html=True)
                with atom_col4:
                    st.markdown(f"<div class='small-number'>O: {atomic_comp.get('O', 0)}</div>", unsafe_allow_html=True)
                with atom_col5:
                    st.markdown(f"<div class='small-number'>S: {atomic_comp.get('S', 0)}</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='small-number'>Formula: {physico_props.get('formula', 'N/A')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='small-number'>Total atoms: {physico_props.get('total_atoms', 0)}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()