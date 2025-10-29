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

# Try to import RDKit (optional)
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
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

# Custom CSS
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #e0f2fe 0%, #fae8ff 100%);}
    .stButton>button {
        width: 100%; background: linear-gradient(90deg, #9333ea 0%, #ec4899 100%);
        color: white; font-weight: bold; border: none; padding: 0.75rem;
        border-radius: 0.75rem; font-size: 1.1rem;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(147, 51, 234, 0.4);
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
        self.aa_weights = {
            'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165,
            'G': 75, 'H': 155, 'I': 131, 'K': 146, 'L': 131,
            'M': 149, 'N': 132, 'P': 115, 'Q': 146, 'R': 174,
            'S': 105, 'T': 119, 'V': 117, 'W': 204, 'Y': 181
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
        # Note: The original features.py used list concatenation,
        # but the app.py had a bug. This is the correct logic
        # from the training notebook (33 + 17 = 50 features).
        return protein_features + drug_features


# ==================== REAL ML PREDICTOR ====================
class RealMLPredictor:
    """Uses actual trained ML models"""

    def __init__(self):
        self.models_loaded = False

        # --- MODIFIED ---
        # We ALWAYS use the local FeatureExtractor class defined in this file.
        # Loading the .pkl is problematic due to module path conflicts from Jupyter.
        self.feature_extractor = FeatureExtractor()
        # --- END MODIFICATION ---

        self.load_models()

    def load_models(self):
        """Load trained models from the 'models/' directory"""
        # This script is in src/, so models are at ../models
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        model_dir = os.path.abspath(model_dir)

        try:
            with open(os.path.join(model_dir, 'classifier.pkl'), 'rb') as f:
                self.classifier = pickle.load(f)
            with open(os.path.join(model_dir, 'regressor.pkl'), 'rb') as f:
                self.regressor = pickle.load(f)
            with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)

            # --- REMOVED ---
            # We no longer load the feature_extractor.pkl file
            # --- END REMOVED ---

            self.models_loaded = True
            return True

        except FileNotFoundError:
            # self.feature_extractor = FeatureExtractor() # Already created in __init__
            return False

    # === NEW DEBUG VERSION OF PREDICT ===
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
            # Ensure it's a NumPy array first
            X = np.array(combined)
            # Check if it's already 2D, if not, reshape
            if X.ndim == 1:
                X = X.reshape(1, -1)
            # --- Explicitly convert to float32 ---
            X = X.astype(np.float32)
            print(f"  Combined feature vector shape: {X.shape}, dtype: {X.dtype}")

            if self.models_loaded:
                # 4a. Scale
                print("[Step 4a] Scaling features...")
                X_scaled = None # Initialize
                try:
                    print(f"  Input to scaler - Shape: {X.shape}, Type: {X.dtype}") # DEBUG
                    X_scaled = self.scaler.transform(X)
                     # --- Explicitly convert scaled output ---
                    X_scaled = X_scaled.astype(np.float32)
                    print(f"  Output from scaler - Shape: {X_scaled.shape}, Type: {X_scaled.dtype}") # DEBUG
                except Exception as e_scale:
                    print(f"--- CRASH POINT: Scaling Failed ---")
                    print(f"Error: {e_scale}")
                    st.error(f"Error during feature scaling: {e_scale}")
                    import traceback; traceback.print_exc() # Print full traceback
                    return None

                # 4b. Predict Probability (Classifier)
                print("[Step 4b] Predicting binding probability (Classifier)...")
                binding_prob = None # Initialize
                try:
                    print(f"  Input to classifier.predict_proba - Shape: {X_scaled.shape}, Type: {X_scaled.dtype}") # DEBUG
                    # Ensure input is float32
                    binding_prob_array = self.classifier.predict_proba(X_scaled.astype(np.float32))
                    binding_prob = binding_prob_array[0][1] # Get prob of class '1' (Binds)
                    print(f"  Predicted binding probability: {binding_prob}")
                except Exception as e_class:
                    print(f"--- CRASH POINT: Classifier Prediction Failed ---")
                    print(f"Error: {e_class}")
                    st.error(f"Error during classification prediction: {e_class}")
                    import traceback; traceback.print_exc() # Print full traceback
                    return None

                # 4c. Predict IC50 (Regressor)
                print("[Step 4c] Predicting IC50 (Regressor)...")
                ic50_value = None # Initialize
                try:
                    print(f"  Input to regressor.predict - Shape: {X_scaled.shape}, Type: {X_scaled.dtype}") # DEBUG
                     # Ensure input is float32
                    ic50_raw = self.regressor.predict(X_scaled.astype(np.float32))
                    ic50_value = ic50_raw[0]
                    print(f"  Predicted raw IC50: {ic50_value}")
                    ic50_value = max(0.0, float(ic50_value)) # Clamp and ensure float
                    print(f"  Predicted clamped IC50: {ic50_value}")
                except Exception as e_reg:
                    print(f"--- CRASH POINT: Regressor Prediction Failed ---")
                    print(f"Error: {e_reg}")
                    st.error(f"Error during regression prediction: {e_reg}")
                    import traceback; traceback.print_exc() # Print full traceback
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

            result_dict = {
                'binding_probability': float(binding_prob), # Ensure float type
                'ic50_value': float(ic50_value), # Ensure float type
                'classification': classification,
                'protein_length': len(protein_seq),
                'molecular_weight': mol_weight,
                'aromaticity': aromaticity,
                'confidence': confidence,
                'model_type': 'Trained ML Model' if self.models_loaded else 'Demo Mode'
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
    # === END NEW DEBUG VERSION ===


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

    st.markdown('<div style="text-align:center;padding:2rem 0;"><h1 style="font-size:3rem;background:linear-gradient(90deg,#9333ea 0%,#ec4899 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🧬 Drug-Target Interaction Predictor</h1><p style="font-size:1.2rem;color:#6b7280;">ML-Based Prediction of Binding Affinity and Interaction Probability</p></div>', unsafe_allow_html=True)

    if predictor.models_loaded:
        st.success("✅ **Real ML Models Loaded** - Using trained classifier and regressor from `models/` directory")
    else:
        st.warning("⚠️ **Demo Mode** - Models not found in `models/`. Run `train.ipynb` first to generate them.")

    # Sidebar
    with st.sidebar:
        st.header("🧪 Example Data")
        for name, data in EXAMPLES.items():
            if st.button(name, key=f"example_{name}"):

                # --- CHANGED ---
                # These keys now match the widget keys
                st.session_state.protein_input = data['protein']
                st.session_state.smiles_input = data['smiles']
                # --- END CHANGED ---

                st.session_state.results = None
                st.rerun()
        st.markdown("---")
        st.info("💡 **How It Works:**\n\nUses a trained XGBoost model to predict if a drug (SMILES) will bind to a protein (sequence) and estimates the binding affinity.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🧫 Protein Target Sequence")

        # --- CHANGED ---
        # Removed 'value' parameter and set 'key' to match the session state
        protein_seq = st.text_area(
            "Enter amino acid sequence",
            height=120,
            key="protein_input" # <-- Key now matches button
        )
        # --- END CHANGED ---

        st.markdown("### 💊 Drug Molecule (SMILES)")

        # --- CHANGED ---
        # Removed 'value' parameter and set 'key' to match the session state
        drug_smiles = st.text_input(
            "Enter SMILES notation",
            key="smiles_input" # <-- Key now matches button
        )
        # --- END CHANGED ---

        c1, c2 = st.columns(2)
        with c1:
            predict_btn = st.button("🔮 Predict Interaction")
        with c2:
            reset_btn = st.button("🔁 Reset")

        if reset_btn:
            # This logic now works perfectly
            st.session_state.protein_input = ''
            st.session_state.smiles_input = ''
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
                    # The lines causing the previous crash were removed here


    if st.session_state.get('results'):
        results = st.session_state.results

        # Display molecule visualization in the second column
        with col2:
            if RDKIT_AVAILABLE:
                st.markdown("### 🧩 Molecule Visualization")
                mol_smiles = st.session_state.get('smiles_input', '')
                if mol_smiles:
                    try:
                        mol = Chem.MolFromSmiles(mol_smiles)
                        if mol:
                            st.image(Draw.MolToImage(mol, size=(400, 300)), caption="2D Molecular Structure")
                        else:
                            st.warning("Could not generate image for this SMILES string.")
                    except Exception as img_err:
                        # Catch specific image generation errors
                        print(f"RDKit image generation error: {img_err}") # Debug print
                        st.warning("RDKit error generating molecule image.")

        # Display results in the main column
        with col1:
            st.markdown("---")
            st.markdown("## 🧾 Prediction Results")
            st.caption(f"Model type: {results['model_type']}")

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Binding Probability", f"{results['binding_probability']*100:.1f}%")
            res_col2.metric("Predicted IC50 (μM)", f"{results['ic50_value']:.2f}")
            res_col3.metric("Classification", results['classification'])

            st.markdown("### ⚙️ Protein Properties (Analyzed)")
            prop_col1, prop_col2, prop_col3 = st.columns(3)
            prop_col1.metric("Length", f"{results['protein_length']} aa")
            prop_col2.metric("Mol. Weight", f"~{results['molecular_weight']:.0f} Da")
            prop_col3.metric("Aromaticity", f"{results['aromaticity']*100:.1f}%")


if __name__ == "__main__":
    main()