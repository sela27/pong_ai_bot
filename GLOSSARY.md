# Pong AI Bot

A locally built Pong game and a learning player that teaches itself to play it from the screen image and the score alone, in the spirit of DeepMind's DQN.

## Players

**Agent**:
The learning player. It knows nothing about balls or paddles; it sees only Observations and Rewards.
_Avoid_: AI, bot, model, network

**Opponent**:
The non-learning player on the other side of the court, whose play follows fixed rules.
_Avoid_: CPU, enemy, computer player

**Human**:
A person who moves a Paddle with the keyboard.
_Avoid_: User, player, person

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

**Observation**:
The image of the game screen given to the Agent at each step; together with the Reward, it is all the Agent knows about the game.
_Avoid_: State, frame (when meaning what the Agent sees)

**Reward**:
The change in score the Agent receives after a step: positive when it wins a Point, negative when it loses one.
_Avoid_: Score (when meaning the per-step signal)

## Measuring

**Score margin**:
The Agent's Points minus the Opponent's Points at the end of a Game, from -21 to +21.
_Avoid_: Result, win rate

**Evaluation**:
A set of 100 Games in which the Agent plays its best known move every step, with no random exploration, used to judge how well it plays.
_Avoid_: Test run, benchmark
