# Pong AI Bot

A locally built Pong game and a learning player that teaches itself to play it from the screen image and the score alone, in the spirit of DeepMind's DQN.

## Players

**Agent**:
The learning player. It knows nothing about balls or paddles; it sees only Observations and Rewards.
_Avoid_: AI, bot, model, network

**Opponent**:
The non-learning player the Agent plays against, on the other side of the Court. Its rules are the same in every Game, so results stay comparable.
_Avoid_: CPU, enemy, computer player

**Human**:
A person who moves a Paddle with the keyboard.
_Avoid_: User, player, person

**Scripted player**:
A non-learning player that plays in the Agent's place by hand-written rules, used to measure what can be achieved against the Opponent.
_Avoid_: Baseline, bot, dummy

## Court

**Court**:
The rectangular playing area, closed by walls along the top and bottom and open at each player's end.
_Avoid_: Field, board, table, arena

**Paddle**:
The bar a player moves up and down along its own end of the Court to return the ball.
_Avoid_: Bat, racket, player (when meaning the bar itself)

## Play

**Game**:
One match between two players, one on each side of the Court, ending when either side reaches 21 Points.
_Avoid_: Match, round

**Point**:
One unit of score, won when the ball leaves the Court through the other side's open end.
_Avoid_: Goal

**Frame**:
The smallest tick of game time. The ball and the Paddles move once every Frame.
_Avoid_: Tick

**Action**:
One of the three choices of a player in the Agent's place: move its Paddle up, move it down, or keep it still.
_Avoid_: Move, command, key

**Step**:
One decision of the Agent or a Scripted player: it chooses an Action, the Action is held for a fixed number of Frames, and then it receives the next Observation and the Reward.
_Avoid_: Turn, tick, frame

**Observation**:
The image of the Court given to the Agent at each Step. The Agent decides from its last four Observations; together with the Reward, they are all it knows about the Game.
_Avoid_: State, frame (when meaning what the Agent sees), screen

**Reward**:
The change in score the Agent receives after a Step: positive when it wins a Point, negative when it loses one.
_Avoid_: Score (when meaning the per-step signal)

## Measuring

**Score margin**:
The Agent's Points minus the Opponent's Points at the end of a Game, from -21 to +21.
_Avoid_: Result, win rate

**Evaluation**:
A set of 100 Games in which the Agent chooses its best known Action every Step, with no random exploration, used to judge how well it plays.
_Avoid_: Test run, benchmark
