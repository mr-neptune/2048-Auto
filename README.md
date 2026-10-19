# 2048-Auto

2048 is a well known puzzle game, where the player will move number tiles in a 4×4 grid to merge the number 2048, or even higher.

The game logic is implemented in Python 3, and the user interface is implemented in HTML, CSS, and JavaScript. A small Flask server (`app.py`) exposes HTTP endpoints so the same environment can drive both the browser UI and automated agents.

## Demo

<p align="center">
  <img src="docs/images/demo_2048.gif" width="600" alt="The afterstate CNN agent auto-playing 2048 in the browser UI, finishing a game with a 2048 tile at 26,964 points">
</p>

The final moves of a game played by the afterstate CNN agent (episode-3000 checkpoint, no randomness): it builds a 2048 tile and ends at 26,964 points. To watch it yourself, run `python -m flask --app app run --port 5001`, open http://localhost:5001, pick **CNN Afterstate (new)**, and press **Start Auto Play**.

## CNN afterstate agent

`CNN/cnn_afterstate.py` is the current best agent. It replaced the earlier DQN (`CNN/cnn_agent.py`), which stalled at about 2.5K points after 20K episodes.

**The idea.** A 2048 move has two parts: your slide, which is fully predictable, and a random new tile, which is luck. The DQN had to guess both at once. The afterstate agent slides the tiles itself (the rules are known), so it only asks the network one question: *how good is this board right after my slide, before the new tile appears?* That board is the **afterstate**.

```mermaid
flowchart LR
    S["Current board"] --> A["Try all 4 slides<br/>(no new tile yet)"]
    A --> N["CNN rates each<br/>afterstate: V"]
    N --> P["Play the move with the highest<br/>points gained + V"]
    P --> G["Game adds a random<br/>2 or 4 tile"]
    G --> S
```

Training details:

- **Reward is the real game score.** There are no hand-made bonus rewards.
- **TD(λ) targets.** After each game, every afterstate is trained toward a mix of the points actually earned afterwards and the network's own estimate of later boards (λ = 0.5, no discount).
- **8× data.** A board is equally good rotated or mirrored, so each training sample gets one of the 8 symmetries at random.
- **No random exploration.** The random tile spawns already make every game different.
- **Replay buffer** of 50K afterstates. Each afterstate is trained on about 8 times.

### Results

Greedy play (always the best move, no random moves). "Reached" means the game's largest tile was at least that tile.

| Reached tile | Old DQN<br/>20K episodes | Afterstate CNN<br/>3K episodes | **Afterstate CNN**<br/>**5K episodes** |
|---|---|---|---|
| 512 | 3.5% | 90% | **92.5%** |
| 1024 | 0% | 70% | **73.5%** |
| 2048 | 0% | 18.5% | **29.5%** |
| 4096 | 0% | 0% | 0% |
| Average score | 2.5K | 14.5K | **16.9K** |
| Best score | – | 36.2K | **36.7K** |
| Games tested | 200 | 200 | 200 |

With a quarter of the old DQN's training episodes, the afterstate agent reaches 2048 in about 30% of games. 4096 has appeared 11 times during training (first at episode 2,744, best score 57K), but not yet in a 200-game test.

### Training progress

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/training_progress_dark.png">
  <img src="docs/images/training_progress_light.png" alt="Three charts over 5,000 training episodes: average score rising to about 17–19K, share of games reaching 1024 rising to about 80% and 2048 to about 30%, and training loss rising from 0.004 to about 0.07">
</picture>

- **Score.** Each dot is one game and the line is the 100-game average. Progress plateaus around episodes 800–2,300, climbs to about 17K by episode 3,500, and then levels off.
- **Tile rates.** These show the share of the last 500 games that reached 1024, 2048 and 4096. Since episode 3,500 they hold at about 80% for 1024 and 30% for 2048.
- **Loss going up is expected here.** The network predicts the points still to come, and those values grow as the agent survives longer (a 2048 game is worth roughly 4× a 512 game). Bigger targets mean bigger squared errors, even when the relative error is flat. The spike near episode 4,800 lines up with a short score dip, and both recovered on their own.

Regenerate the charts with `python -m CNN.plot_training` (needs `matplotlib`).

### Network architecture

```mermaid
flowchart TD
    IN["Board · 4×4 one-hot, 16 channels<br/>(empty, 2, 4, 8, … 32768)"]
    IN --> H1["Conv 1×2 · 128 · ReLU<br/>4×3 map"]
    IN --> V1["Conv 2×1 · 128 · ReLU<br/>3×4 map"]
    H1 --> H2["Conv 1×2 · 128<br/>4×2 map"]
    H1 --> X1["Conv 2×1 · 128<br/>3×3 map"]
    V1 --> X2["Conv 1×2 · 128<br/>3×3 map"]
    V1 --> V2["Conv 2×1 · 128<br/>2×4 map"]
    H1 & H2 & X1 & X2 & V2 & V1 --> CAT["Flatten + concatenate all 6 maps<br/>58 × 128 = 7,424 features"]
    CAT --> FC["Dense 256 · ReLU"]
    FC --> OUT["V(afterstate) · one number"]
```

- **Input.** Each cell becomes 16 on/off channels: channel 0 means empty and channel *k* means tile 2^k. This way the network sees "a 512 next to a 512" as a pattern, not as a large number.
- **Directional convolutions.** 1×2 kernels look at horizontal neighbor pairs and 2×1 kernels look at vertical pairs. Those are exactly the pairs that can merge on a left/right or up/down slide. Stacking two layers covers rows and columns of 3, and the cross branches (1×2 then 2×1, and the reverse) cover 2×2 squares.
- **Head.** All six feature maps are flattened, concatenated and passed through one hidden layer to a single value. That value is the predicted future score (scaled by 1/1000) of the afterstate.
- **Size.** 2.04M parameters: 1.90M in the first dense layer and 140K in the convolutions. This is the same trunk the old DQN used. Only the output changed, from 4 move values to 1 board value.

### Train, resume, evaluate

```bash
# train (CPU is ~2x faster than the Metal GPU on Apple silicon; wrap in caffeinate so the Mac stays awake)
caffeinate -i python -m CNN.cnn_afterstate --episodes 5000 --resume \
  --save-path bin/cnn_after_pilot.chkpt --metrics-path CNN/logs/cnn_after_pilot_train.csv \
  --log-interval 100 --checkpoint-interval 200 >> CNN/logs/cnn_after_pilot.out 2>&1

# evaluate a checkpoint with 200 greedy games (prints the tile table)
python -m CNN.cnn_afterstate --eval-only --resume --save-path bin/cnn_after_pilot.chkpt --eval-episodes 200
```

`--resume` continues from the last checkpoint: weights, optimizer state, replay buffer and episode counter. Checkpoints are written atomically every `--checkpoint-interval` episodes, so an interrupted run loses at most that many episodes. On an M2 Pro CPU, 3,000 episodes take about 1.2 hours.

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
