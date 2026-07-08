# ForgeMind — Team Workflow (ROOKIE GUIDE)

Read this once and you'll know exactly how 7–10 people build one project
without stepping on each other.

----------------------------------------------------------------------
## THE BIG IDEA (read this first)
----------------------------------------------------------------------

  GitHub repo  =  THE BRAIN. One master copy of all code. Source of truth.
  SSH box      =  A GPU ENGINE. We only use it to RUN heavy stuff
                  (download datasets, train models, run the demo).
  Your laptop  =  WHERE YOU WRITE CODE. Your own machine, your own copy.

You do NOT all edit files directly on the box. If 10 people log into one
box and edit the same files, the last person to save wins and everyone
else's work vanishes. Git stops that: everyone has their own copy, and
git merges changes safely.

----------------------------------------------------------------------
## SETUP — do this ONCE on your laptop
----------------------------------------------------------------------

1. Install git + Python 3.10+.
2. Clone the repo:
       git clone https://github.com/IMMORTALxTHOUGHT/AGIHUNTERS_PS2.git
       cd AGIHUNTERS_PS2
3. Make your personal branch (use your name):
       git checkout -b alice
4. Make a local venv (CPU is fine for MOST modules):
       python -m venv venv && source venv/bin/activate
       pip install -r requirements.txt

----------------------------------------------------------------------
## DAILY LOOP — what you do every session
----------------------------------------------------------------------

  1. git checkout main
  2. git pull origin main            # get everyone's latest merged work
  3. git checkout -b alice           # or: git checkout alice (if it exists)
  4. ... edit YOUR module's files ...
  5. git add <file>                  # only add files you changed
  6. git commit -m "short note of what you did"
  7. git push -u origin alice
  8. Open a Pull Request (PR) on GitHub. Lead reviews and merges into main.

----------------------------------------------------------------------
## RULES — so we don't break each other's work
----------------------------------------------------------------------

  * NEVER push directly to main. Always use a branch + PR.
  * Only touch the files assigned to YOUR role (see below). Need to change
    someone else's file? Talk to them first.
  * Pull main before you start each session.
  * Test your own module before pushing (a small test or `python -m x`).
  * Adding a pip package? Tell the lead so requirements.txt stays clean.

----------------------------------------------------------------------
## ROLE SPLIT — who owns what (collapses to fit 7–10 people)
----------------------------------------------------------------------

  A. Data + Detection (PatchCore)   -> data/loaders.py, models/patchcore.py
  B. Classification (ViT)           -> models/vit_classifier.py, models/embedder.py
  C. Memory (FAISS)                 -> storage/faiss_store.py
  D. Knowledge Graph                -> storage/knowledge_graph.py
  E. Agents (debate + moderator)    -> agents/debate.py, agents/moderator.py
  F. Analytics (DNA / calib / health) -> analytics/*.py
  G. Self-learning                  -> pipeline self-learn wiring
  H. Dashboard                      -> dashboard/app.py
  I. Orchestrator                   -> pipeline/run.py
  J. Docs                           -> ARCHITECTURE.md, TEAM.md, README.md

  One person can own two small roles. The point: each file has ONE owner,
  so merges rarely conflict.

----------------------------------------------------------------------
## ON THE SSH BOX — GPU runner only
----------------------------------------------------------------------

The LEAD (or one designated runner) does this. Only ONE person runs
commands on the box at a time — announce in group chat first.

      cd /DATA/AGIHUNTERS_PS2
      git pull origin main
      source hackathon/bin/activate
      python -m pipeline.run --image <test.jpg>     # run the demo
      # or train:
      python -m training.train_patchcore

The box already has: datasets/ (mvtec, neu, dagm), models/qwythos/,
and the 'hackathon' venv. Those are gitignored — don't touch them,
just pull the new code on top.

----------------------------------------------------------------------
## IF YOU HAVE NO LAPTOP GPU (must work on the box)
----------------------------------------------------------------------

Each person clones into THEIR OWN folder so you never share a directory:

      git clone https://github.com/IMMORTALxTHOUGHT/AGIHUNTERS_PS2.git ~/dev_alice
      cd ~/dev_alice && git checkout -b alice
      # ... edit, commit, push branch, open PR ...

Never share a folder with another person. Push your branch, open a PR,
lead merges. The box is for RUNNING, not for shared editing.

----------------------------------------------------------------------
## EMERGENCIES
----------------------------------------------------------------------

  * Merge conflict? Don't panic. Git shows the file with ==== markers.
    Keep the correct version, delete the markers, `git add`, `git commit`.
  * Lost work? `git reflog` shows your history. Nothing is truly gone
    for 30 days.

----------------------------------------------------------------------
## QUICK COMMAND CHEAT-SHEET
----------------------------------------------------------------------

  git clone <url>            # first time only
  git checkout main          # switch to main
  git pull origin main       # get latest
  git checkout -b myname     # new personal branch
  git add file.py            # stage a change
  git commit -m "msg"        # save it
  git push -u origin myname  # send to GitHub
  # then: click "New Pull Request" on GitHub

That's it. Clone -> branch -> edit -> commit -> push -> PR -> lead merges.
