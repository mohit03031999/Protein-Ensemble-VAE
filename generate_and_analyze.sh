#!/bin/bash

# Complete pipeline for generating and analyzing ensemble structures
# Usage: ./generate_and_analyze.sh <checkpoint_path> <data_manifest>

set -e  # Exit on error

echo "🧬 PROTEIN VAE ENSEMBLE GENERATION & ANALYSIS PIPELINE"
echo "========================================================"
echo ""

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <checkpoint_path> <data_manifest> [num_samples] [output_dir]"
    echo ""
    echo "Example:"
    echo "  $0 checkpoints/single_protein.pt protein_ensemble_dataset/manifest_single_val.csv 20 results"
    echo ""
    exit 1
fi

CHECKPOINT=$1
DATA_MANIFEST=$2
NUM_SAMPLES=${3:-10}  # Default: 10 samples
OUTPUT_DIR=${4:-generated_pdbs}
ANALYSIS_DIR="${OUTPUT_DIR}_analysis"

echo "Configuration:"
echo "  Checkpoint:    $CHECKPOINT"
echo "  Data:          $DATA_MANIFEST"
echo "  Num samples:   $NUM_SAMPLES"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Analysis dir:  $ANALYSIS_DIR"
echo ""

# Step 1: Generate ensembles
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Generating Ensemble Structures"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Note: Model hyperparameters (z_global, z_local, d_model, etc.) are now
# automatically loaded from the checkpoint, so you don't need to specify them!
python generate_ensemble_pdbs.py \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA_MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --use_seqemb \
    --device cuda

echo ""
echo "✅ Ensemble generation complete!"
echo ""

# Step 2: Analyze ensembles
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Analyzing Ensemble Quality"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if matplotlib is available
if python -c "import matplotlib" 2>/dev/null; then
    python analyze_ensemble.py \
        --pdb_dir "$OUTPUT_DIR" \
        --output_dir "$ANALYSIS_DIR"
    
    echo ""
    echo "✅ Analysis complete!"
    echo ""
    echo "📊 Results:"
    echo "  - PDB files:      $OUTPUT_DIR/"
    echo "  - Analysis plots: $ANALYSIS_DIR/"
    echo "  - Summary:        $OUTPUT_DIR/generation_summary.txt"
    echo "  - Detailed:       $ANALYSIS_DIR/detailed_analysis.txt"
    echo "  - Parameters:     ${OUTPUT_DIR}_parameters/"
else
    echo "⚠️  matplotlib not found - skipping visualization"
    echo "   Install with: pip install matplotlib seaborn"
    echo ""
    echo "📊 Results:"
    echo "  - PDB files: $OUTPUT_DIR/"
    echo "  - Summary:   $OUTPUT_DIR/generation_summary.txt"
fi

# Step 3: Visualize model parameters
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Visualizing Model Parameters (Weights & Biases)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PARAM_DIR="${OUTPUT_DIR}_parameters"

if python -c "import matplotlib" 2>/dev/null; then
    python visualize_parameters.py \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$PARAM_DIR" \
        --z_global 512 \
        --z_local 256 \
        --device cpu
    
    echo ""
    echo "✅ Parameter visualization complete!"
    echo "  - Parameter plots: $PARAM_DIR/"
    echo "  - Summary:         $PARAM_DIR/parameter_summary.txt"
else
    echo "⚠️  matplotlib not found - skipping parameter visualization"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "VISUALIZATION TIPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "PyMOL commands:"
echo "  cd $OUTPUT_DIR"
echo "  pymol struct_000_ground_truth.pdb struct_000_reconstruction.pdb struct_000_ensemble.pdb"
echo ""
echo "In PyMOL console:"
echo "  align reconstruction, ground_truth"
echo "  color red, ground_truth"
echo "  color green, reconstruction"
echo "  color cyan, ensemble"
echo "  set all_states, on  # Show all ensemble models"
echo ""
echo "VMD commands:"
echo "  vmd -m $OUTPUT_DIR/struct_000_*.pdb"
echo ""
echo "Chimera commands:"
echo "  chimera $OUTPUT_DIR/struct_000_ground_truth.pdb $OUTPUT_DIR/struct_000_ensemble.pdb"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Pipeline complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

