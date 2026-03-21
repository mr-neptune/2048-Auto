# 2048-Auto

2048 is a well known puzzle game, where the player will move number tiles in a 4×4 grid to merge the number 2048, or even higher.

The game logic is implemented in Python 3, and the user interface is implemented in HTML, CSS, and JavaScript. A small Flask server (`app.py`) exposes HTTP endpoints so the same environment can drive both the browser UI and automated agents.

## Local training with Q-learning

The repository now ships with a lightweight Q-learning trainer that plays directly against the Python environment. It collects experience, updates a tabular policy, and can optionally resume from a saved checkpoint.

Run a training session (this also evaluates the greedy policy once training finishes):

```bash
python train_agent.py --episodes 5000 --log-interval 250
```

Key options:

- `--episodes` – number of training games to run.
- `--alpha`, `--gamma`, `--epsilon`, `--epsilon-decay`, `--epsilon-min` – standard Q-learning hyperparameters.
- `--load-path` – path to an existing `bin/q_table.pkl` to continue training.
- `--save-path` – where to store the trained table (defaults to `bin/q_table.pkl`).
- `--eval-episodes` – how many greedy runs to average for the evaluation summary.

After training you can point the front-end’s auto-play logic to the resulting policy or use the saved table to bootstrap more advanced agents.

## Optional Postgres metrics logging

If you want to log per-episode metrics to Postgres (for visual inspection in tools like TablePlus), run Postgres with Docker and pass a connection URL to the training scripts.

```bash
docker compose up -d
export POSTGRES_URL="postgresql://2048_user:2048_pass@localhost:5432/2048_db"
python CNN/cnn_agent.py --episodes 1000 --db-run-name "cnn-run-1"
python MLP/mlp_agent.py --episodes 1000 --db-run-name "mlp-run-1"
```

Schema reference lives at `db/schema.sql`.
