"""
PROGRAM: 3x3 Q-Learning Demonstration with Visual Q-Values and Action Arrows
AUTHOR: Dr. Aliasghar Khavasi
DATE: 2026 - January

DESCRIPTION:
A Tkinter-based GUI program that demonstrates tabular Q-learning in a small,
fully observable grid-world:

1) Environment (3×3 grid):
   - Discrete states: 0..8 (row-major indexing)
   - Start state: top-left (0,0) → state 0
   - Goal state: bottom-right (2,2) → reward = 1.0
   - Other transitions: reward = 0.0
   - Optional barriers via BARRIER_CELLS (rendered as hatched/gray cells)

2) Q-learning core:
   - Q-table shape: (9 states × 4 actions), initialized to zeros
   - Actions: up, right, down, left (epsilon-greedy selection)
   - Update rule:
       Q(s,a) ← (1−α)Q(s,a) + α[r + γ max_a' Q(s',a')]
   - When the goal is reached, the agent resets to the start state

3) Interactive hyperparameters and controls:
   - Spinboxes: Alpha (α), Gamma (γ), Epsilon (ε) updated during training
   - Speed slider controls sleep time per iteration
   - Buttons: Start, Pause, Resume, Step (single-iteration), Stop

4) Visualization:
   - Each grid cell shows four Q-values (up/right/down/left) as text labels
   - Cell background intensity scales with the best Q-value in the cell
   - Agent is shown as a yellow marker in the current cell
   - A directional arrow indicates the last action taken (color-coded by action)

5) Logging (two memos):
   - Memo #1 (Progress Log): high-level training progress per iteration
   - Memo #2 (Detailed Calculations): step-by-step Q-update trace including
     states, action, reward, done flag, and current (α, γ, ε)


LEARNING PATH CONNECTION:

Supports a reinforcement learning session introducing:
    - Markov decision processes in grid worlds
    - Tabular Q-learning, epsilon-greedy exploration, and hyperparameter effects
    - Interpreting a learned policy via Q-values and value-based heatmaps
    - The relationship between numeric updates (logs) and visual behavior (canvas)
"""

import tkinter as tk
import numpy as np
import random
import threading
import time

# ----------------------------------------------
# GLOBALS for the 3x3 environment
# ----------------------------------------------
GRID_SIZE = 3
ACTIONS = ['up', 'right', 'down', 'left']  # 0=up,1=right,2=down,3=left
NUM_ACTIONS = len(ACTIONS)

# Q-Learning hyperparameters (these will be updated from GUI spinboxes)
ALPHA = 0.5
GAMMA = 0.9
EPSILON = 0.1

REWARD_GOAL = 1.0
GOAL_POS = (GRID_SIZE-1, GRID_SIZE-1)  # (2,2)

# If you want to treat certain cells as barriers in the 3x3 environment,
# define them here, e.g. BARRIER_CELLS = {(1,1)}
# By default it's empty => no actual barriers in the environment
BARRIER_CELLS = {(1, 1)}

class QLearningCore:
    """
    Minimal 3x3 environment:
      - States: 0..8
      - Start: state=0 => top-left
      - Goal: state=8 => bottom-right => reward=1
    We keep a Q-table shape (9,4).
    """
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.num_states = GRID_SIZE * GRID_SIZE
        self.num_actions = NUM_ACTIONS
        self.Q_table = np.zeros((self.num_states, self.num_actions))
        self.state = 0  # top-left => (0,0)

    def state_to_rc(self, s):
        return (s // self.grid_size, s % self.grid_size)

    def rc_to_state(self, r, c):
        return r * self.grid_size + c

    def is_goal(self, r, c):
        return (r, c) == GOAL_POS

    def step(self, s, action_idx):
        """
        action: 0=up,1=right,2=down,3=left
        return (next_state, reward, done)
        """
        (r, c) = self.state_to_rc(s)
        if action_idx == 0:   # up
            nr = max(r-1, 0)
            nc = c
        elif action_idx == 1: # right
            nr = r
            nc = min(c+1, self.grid_size-1)
        elif action_idx == 2: # down
            nr = min(r+1, self.grid_size-1)
            nc = c
        else:                 # left
            nr = r
            nc = max(c-1, 0)

        ns = self.rc_to_state(nr, nc)
        if self.is_goal(nr, nc):
            rew = REWARD_GOAL
            done = True
        else:
            rew = 0.0
            done = False
        return (ns, rew, done)

    def pick_action(self, s):
        # epsilon-greedy
        global EPSILON
        if random.random() < EPSILON:
            return random.randint(0, self.num_actions-1)
        else:
            return np.argmax(self.Q_table[s])

    def q_learning_step(self):
        """
        One Q-learning step from self.state
        Return (old_s, action_idx, next_s, oldQ, newQ, rew, done)
        so we can display the update details in memo #2.
        """
        old_s = self.state
        a_idx = self.pick_action(old_s)
        (next_s, rew, done) = self.step(old_s, a_idx)

        global ALPHA, GAMMA
        oldQ = self.Q_table[old_s, a_idx]
        maxQ_next = np.max(self.Q_table[next_s])
        newQ = (1-ALPHA)*oldQ + ALPHA*(rew + GAMMA*maxQ_next)
        self.Q_table[old_s, a_idx] = newQ

        self.state = next_s
        if done:
            self.state = 0  # reset agent to top-left
        return (old_s, a_idx, next_s, oldQ, newQ, rew, done)


class RLGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("3x3 Q-Learning with Hatched Walls & Arrow Directions")

        self.core = QLearningCore()
        self.running = False
        self.paused = False
        self.step_mode = False
        self.training_thread = None

        self.last_action = None  # store last action for arrow color/direction

        self.build_controls()
        self.build_canvas()
        self.build_memos()

    # --------------------------------------------
    # UI: Controls
    # --------------------------------------------
    def build_controls(self):
        top_frame = tk.Frame(self.root, bd=2, relief=tk.RIDGE)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Speed control
        tk.Label(top_frame, text="Speed(sec/iter):").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=0.1)
        tk.Scale(top_frame, variable=self.speed_var,
                 from_=0.0, to=1.0, resolution=0.05,
                 orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=5)

        # Hyperparameter spinboxes
        # 1) ALPHA
        tk.Label(top_frame, text="Alpha:").pack(side=tk.LEFT, padx=5)
        self.alpha_var = tk.DoubleVar(value=0.5)
        tk.Spinbox(top_frame, textvariable=self.alpha_var, from_=0.0, to=1.0, increment=0.05,
                   width=5).pack(side=tk.LEFT)

        # 2) GAMMA
        tk.Label(top_frame, text="Gamma:").pack(side=tk.LEFT, padx=5)
        self.gamma_var = tk.DoubleVar(value=0.9)
        tk.Spinbox(top_frame, textvariable=self.gamma_var, from_=0.0, to=1.0, increment=0.05,
                   width=5).pack(side=tk.LEFT)

        # 3) EPSILON
        tk.Label(top_frame, text="Epsilon:").pack(side=tk.LEFT, padx=5)
        self.epsilon_var = tk.DoubleVar(value=0.1)
        tk.Spinbox(top_frame, textvariable=self.epsilon_var, from_=0.0, to=1.0, increment=0.05,
                   width=5).pack(side=tk.LEFT)

        # Control buttons
        tk.Button(top_frame, text="Start", command=self.on_start).pack(side=tk.LEFT, padx=2)
        tk.Button(top_frame, text="Pause", command=self.on_pause).pack(side=tk.LEFT, padx=2)
        tk.Button(top_frame, text="Resume", command=self.on_resume).pack(side=tk.LEFT, padx=2)
        tk.Button(top_frame, text="Step", command=self.on_step).pack(side=tk.LEFT, padx=2)
        tk.Button(top_frame, text="Stop", command=self.on_stop).pack(side=tk.LEFT, padx=2)

    # --------------------------------------------
    # UI: Canvas
    # --------------------------------------------
    def build_canvas(self):
        canvas_frame = tk.Frame(self.root, bd=2, relief=tk.RIDGE)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, width=300, height=300, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # For each cell, we have a rectangle + 4 small text labels for Q-values
        # Also, we highlight the agent's cell with an oval or 'A'
        self.cell_shapes = {}
        self.agent_marker = None
        self.agent_arrow = None

    def on_canvas_resize(self, event):
        self.build_grid_canvas()

    def build_grid_canvas(self):
        self.canvas.delete("all")
        self.cell_shapes.clear()
        self.agent_marker = None
        self.agent_arrow = None

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        size = min(w,h)
        cell_size = size // GRID_SIZE

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x1 = c*cell_size
                y1 = r*cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                if (r,c) in BARRIER_CELLS:
                    rect_id = self.canvas.create_rectangle(
                        x1,y1,x2,y2,
                        fill="gray", outline="gray",
                        stipple="gray25"
                    )
                else:
                    rect_id = self.canvas.create_rectangle(x1,y1,x2,y2, fill="white", outline="gray")

                # text for Q-up => near top
                up_y = y1 + cell_size*0.15
                up_x = (x1+x2)//2
                up_id = self.canvas.create_text(up_x, up_y, text="0.00", fill="black", font=("Arial",8,"bold"))

                # Q-right => near right
                rt_x = x2 - cell_size*0.15
                rt_y = (y1+y2)//2
                rt_id = self.canvas.create_text(rt_x, rt_y, text="0.00", fill="black", font=("Arial",8,"bold"))

                # Q-down => near bottom
                dn_y = y2 - cell_size*0.15
                dn_x = (x1+x2)//2
                dn_id = self.canvas.create_text(dn_x, dn_y, text="0.00", fill="black", font=("Arial",8,"bold"))

                # Q-left => near left
                lt_x = x1 + cell_size*0.15
                lt_y = (y1+y2)//2
                lt_id = self.canvas.create_text(lt_x, lt_y, text="0.00", fill="black", font=("Arial",8,"bold"))

                self.cell_shapes[(r,c)] = {
                    "rect": rect_id,
                    "up": up_id,
                    "right": rt_id,
                    "down": dn_id,
                    "left": lt_id,
                    "coords": (x1,y1,x2,y2)
                }

        # Mark start & goal with colored outlines
        s_rect = self.cell_shapes[(0,0)]["rect"]
        self.canvas.itemconfig(s_rect, outline="red", width=2)
        g_rect = self.cell_shapes[(GRID_SIZE-1,GRID_SIZE-1)]["rect"]
        self.canvas.itemconfig(g_rect, outline="green", width=2)

        self.update_canvas_colors()
        self.update_agent_position()

    def update_canvas_colors(self):
        if not self.cell_shapes:
            return
        Q = self.core.Q_table
        max_q = np.max(Q)
        if max_q <= 0:
            max_q = 1.0

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if (r,c) in BARRIER_CELLS:
                    # skip color scaling, it's hatched
                    continue

                s = r*GRID_SIZE + c
                q_up = Q[s,0]
                q_rt = Q[s,1]
                q_dn = Q[s,2]
                q_lt = Q[s,3]
                bestQ = max(q_up,q_rt,q_dn,q_lt)

                # color scale: white->blue
                ratio = bestQ / max_q
                if ratio<0: ratio=0
                if ratio>1: ratio=1
                blueVal = int(255*ratio)
                color = f"#{0:02x}{0:02x}{blueVal:02x}"

                rect_id = self.cell_shapes[(r,c)]["rect"]
                self.canvas.itemconfig(rect_id, fill=color, stipple="")

                brightness = 0.114*blueVal
                txt_color = "white" if brightness<128 else "black"

                self.canvas.itemconfig(self.cell_shapes[(r,c)]["up"],    text=f"{q_up:.2f}", fill=txt_color)
                self.canvas.itemconfig(self.cell_shapes[(r,c)]["right"], text=f"{q_rt:.2f}", fill=txt_color)
                self.canvas.itemconfig(self.cell_shapes[(r,c)]["down"],  text=f"{q_dn:.2f}", fill=txt_color)
                self.canvas.itemconfig(self.cell_shapes[(r,c)]["left"],  text=f"{q_lt:.2f}", fill=txt_color)

    def update_agent_position(self):
        """Draws/updates a small circle in the agent's cell + an arrow for the last action."""
        if not self.cell_shapes:
            return

        if self.agent_marker:
            self.canvas.delete(self.agent_marker)
            self.agent_marker = None
        if self.agent_arrow:
            self.canvas.delete(self.agent_arrow)
            self.agent_arrow = None

        s = self.core.state
        r = s // GRID_SIZE
        c = s % GRID_SIZE
        (x1,y1,x2,y2) = self.cell_shapes[(r,c)]["coords"]

        cx = (x1+x2)//2
        cy = (y1+y2)//2
        rad = (x2-x1)*0.2

        # circle
        self.agent_marker = self.canvas.create_oval(cx-rad, cy-rad, cx+rad, cy+rad,
                                                    fill="yellow", outline="black")

        # arrow: 0=up=>red,1=right=>blue,2=down=>green,3=left=>purple
        if self.last_action is not None:
            arrow_colors = {0:"red", 1:"blue", 2:"green", 3:"purple"}
            clr = arrow_colors.get(self.last_action, "black")

            if self.last_action == 0:   # up
                self.agent_arrow = self.canvas.create_line(cx, cy, cx, cy-15,
                    arrow=tk.LAST, fill=clr, width=2)
            elif self.last_action == 1: # right
                self.agent_arrow = self.canvas.create_line(cx, cy, cx+15, cy,
                    arrow=tk.LAST, fill=clr, width=2)
            elif self.last_action == 2: # down
                self.agent_arrow = self.canvas.create_line(cx, cy, cx, cy+15,
                    arrow=tk.LAST, fill=clr, width=2)
            else:                       # left
                self.agent_arrow = self.canvas.create_line(cx, cy, cx-15, cy,
                    arrow=tk.LAST, fill=clr, width=2)

    # --------------------------------------------
    # Two Memos: memo1 + memo2
    # --------------------------------------------
    def build_memos(self):
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # left: Memo #1 => "Progress Log"
        self.memo1_frame = tk.LabelFrame(bottom_frame, text="Memo #1: Progress Log")
        self.memo1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.memo1_text = tk.Text(self.memo1_frame, wrap=tk.WORD, width=40, height=15)
        self.memo1_text.pack(fill=tk.BOTH, expand=True)

        # right: Memo #2 => "Detailed Calculations"
        self.memo2_frame = tk.LabelFrame(bottom_frame, text="Memo #2: Detailed Calculations")
        self.memo2_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.memo2_text = tk.Text(self.memo2_frame, wrap=tk.WORD, width=40, height=15)
        self.memo2_text.pack(fill=tk.BOTH, expand=True)

        # Insert some initial text
        intro = (
            "LEARNING PATH:\n"
            "1) Q-Learning in a 3x3 grid.\n"
            "2) 'S' cell => top-left. 'G' cell => bottom-right => reward=1.\n"
            "3) If you define certain cells in BARRIER_CELLS => they are hatched squares.\n"
            "4) 4 Q-values in each cell => up/right/down/left.\n"
            "5) A colored arrow near the agent => last direction taken.\n"
            "6) Memo #1 => high-level progress. Memo #2 => step-by-step details.\n"
        )
        self.memo1_text.insert(tk.END, intro)

    def log_memo1(self, msg):
        self.memo1_text.insert(tk.END, msg + "\n")
        self.memo1_text.see(tk.END)

    def log_memo2(self, msg):
        self.memo2_text.insert(tk.END, msg + "\n")
        self.memo2_text.see(tk.END)

    # --------------------------------------------
    # Training
    # --------------------------------------------
    def on_start(self):
        if self.training_thread and self.training_thread.is_alive():
            self.log_memo1("Training already running.")
            return
        self.log_memo1("Starting Q-learning training.")
        self.running = True
        self.paused = False
        self.step_mode = False

        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()

    def on_stop(self):
        self.log_memo1("Stopping training.")
        self.running = False

    def on_pause(self):
        if not self.running:
            self.log_memo1("Not running => can't pause.")
            return
        self.log_memo1("Pausing training.")
        self.paused = True

    def on_resume(self):
        if not self.running:
            self.log_memo1("Not running => can't resume.")
            return
        if not self.paused:
            self.log_memo1("Already running => no need to resume.")
            return
        self.log_memo1("Resuming training.")
        self.paused = False

    def on_step(self):
        if not self.running:
            self.log_memo1("Starting in step mode.")
            self.running = True
            self.step_mode = True
            self.paused = False
            self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
            self.training_thread.start()
        else:
            self.log_memo1("Performing one step.")
            self.step_mode = True
            self.paused = False

    def training_loop(self):
        iteration = 0
        STEPS_PER_ITER = 5

        # We’ll link the global hyperparams to the spinboxes each iteration:
        global ALPHA, GAMMA, EPSILON

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            # Refresh hyperparams from spinboxes each iteration
            ALPHA   = self.alpha_var.get()
            GAMMA   = self.gamma_var.get()
            EPSILON = self.epsilon_var.get()

            iteration += 1
            # We'll do STEPS_PER_ITER Q-learning steps each iteration
            for step_i in range(STEPS_PER_ITER):
                (old_s, a_idx, ns, oldQ, newQ, rew, done) = self.core.q_learning_step()
                self.last_action = a_idx  # store for arrow

                # Log the details
                old_rc = self.core.state_to_rc(old_s)
                next_rc = self.core.state_to_rc(ns)
                action_str = ACTIONS[a_idx]
                self.log_memo2(
                    f"Iter={iteration}, step={step_i+1}, oldS={old_s}({old_rc}), A={action_str}, "
                    f"oldQ={oldQ:.2f}->newQ={newQ:.2f}, rew={rew}, done={done}, "
                    f"(Alpha={ALPHA:.2f}, Gamma={GAMMA:.2f}, Epsilon={EPSILON:.2f})"
                )

            self.log_memo1(f"Iteration={iteration}: {STEPS_PER_ITER} steps done. Last action={ACTIONS[a_idx]}, Reward={rew}, Done={done}")
            self.update_canvas_colors()
            self.update_agent_position()

            if done:
                self.log_memo1("Goal reached => agent reset to (0,0).")

            if self.step_mode:
                # after one iteration, pause
                self.paused = True
                self.step_mode = False

            time.sleep(self.speed_var.get())

        self.log_memo1("Training loop ended.")


def main():
    root = tk.Tk()
    app = RLGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
