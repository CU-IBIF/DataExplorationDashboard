#!/bin/bash
# Run the whole dashboard on a cluster node (big RAM / GPU), view it from your laptop.
#   sbatch deploy/slurm/serve_on_node.sbatch
#   then follow the ssh -L line printed in featurescope-<jobid>.out
#SBATCH --job-name=featurescope
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=featurescope-%j.out
set -euo pipefail
module load apptainer 2>/dev/null || true
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
NODE=$(hostname -s)
echo "On your laptop:  ssh -N -L 8765:${NODE}:${PORT} ${USER}@${SLURM_SUBMIT_HOST}"
echo "Then open        http://localhost:8765"
SIF=${FEATURESCOPE_SIF:-$HOME/featurescope-gpu.sif}
apptainer exec --nv "$SIF" featurescope serve --host 0.0.0.0 --port "$PORT" --workspace "${WORKSPACE:-$PWD/projects}"
