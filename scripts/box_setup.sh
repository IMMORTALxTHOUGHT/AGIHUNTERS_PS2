#!/usr/bin/env bash
# =====================================================================
# ForgeMind — ONE-SHOT BOX SETUP
# Run THIS on the remote box (the machine where /DATA/AGIHUNTERS_P2 lives).
# It turns the existing project folder into a git repo, ignores the
# heavy/local stuff (datasets, models/*.gguf, the venv), makes the first
# commit, and links the GitHub remote so the team can collaborate.
#
# The box /DATA folder is the SOURCE OF TRUTH (hackathon requires it).
# GitHub is just a backup + collaboration layer on top.
# =====================================================================
set -e

# >>> CHANGE THIS to your exact project folder name on the box <<<
PROJ=/DATA/AGIHUNTERS_P2
cd "$PROJ"

echo "[1/5] git init in place (folder already has datasets/models/venv)"
git init -q
git branch -M main

echo "[2/5] write .gitignore so we ONLY commit code, not big/local files"
cat > .gitignore <<'EOF'
# big data — already on the box, never commit
datasets/
outputs/
*.png
*.jpg
*.jpeg

# model weights — huge GGUF/bin, already on the box
models/*.gguf
models/**/*.gguf
models/*.bin
models/*.pt
models/*.onnx
*.ckpt

# python venv
hackathon/
venv/
.venv/
__pycache__/
*.pyc

# local db / secrets
*.sqlite
*.db
.env

# os junk
.DS_Store
Thumbs.db
EOF

echo "[3/5] first commit of the existing project structure"
git add -A
git commit -q -m "init: ForgeMind project at /DATA (datasets/models/venv gitignored)" \
  || echo "  (nothing new to commit — already clean)"

echo "[4/5] link GitHub remote (reuse the repo already created)"
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/IMMORTALxTHOUGHT/AGIHUNTERS_PS2.git

echo "[5/5] push to GitHub"
echo "  NOTE: the box is now the source of truth, so we force-push to"
echo "  overwrite the old local-scaffold history on GitHub."
git push -u origin main --force

echo ""
echo "DONE. Project is a git repo at $PROJ and mirrored on GitHub."
echo "Next: each teammate works in their OWN branch (see TEAM.md)."
