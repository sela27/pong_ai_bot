"""The `pong` command: one entry point, with a subcommand for each mode."""

import argparse
import copy
import secrets
import sys
import warnings
from collections.abc import Callable, Mapping
from dataclasses import fields
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from pong_ai_bot.environment import PongEnv
from pong_ai_bot.evaluation import Evaluation, SeedSet
from pong_ai_bot.learning_players import (
    LEARNING_PLAYERS,
    LearningPlayer,
    evaluate_milestone,
    milestone_evaluation,
)
from pong_ai_bot.run_directory import (
    EVALUATIONS_FILE,
    SETTINGS_FILE,
    RunDirectory,
    SavedFile,
)
from pong_ai_bot.run_settings import RunSettings
from pong_ai_bot.scripted_players import (
    SCRIPTED_PLAYERS,
    PongMaker,
    evaluate_scripted_player,
)
from pong_ai_bot.width_choice import choose_width

if TYPE_CHECKING:
    import torch
    from torch import nn

    from pong_ai_bot.evaluation import EnvMaker
    from pong_ai_bot.training import MilestoneEvaluation

MilestoneEvaluationMaker = Callable[["nn.Module", "EnvMaker"], "MilestoneEvaluation"]
"""Makes the Evaluation a Run plays of its Milestones, as `milestone_evaluation`
does, from the Evaluation's own network and the player's environment."""


def main(
    argv: list[str] | None = None,
    *,
    learning_players: Mapping[str, LearningPlayer] = LEARNING_PLAYERS,
    make_milestone_evaluation: MilestoneEvaluationMaker = milestone_evaluation,
    pong: PongMaker = PongEnv,
) -> int:
    """Run `pong` with the command line `argv`, sys.argv's unless given, and give
    its exit code.

    The rest are what it plays, the real ones unless a test gives fakes: the
    learning players by name, the Evaluation of a Run's Milestones, and the
    environment the Scripted players' Evaluation plays.
    """
    parser = argparse.ArgumentParser(
        prog="pong",
        description="A locally built Pong game and a DQN Agent that learns to play it.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    commands.add_parser(
        "play",
        help="play as the Human against the Opponent, in a window",
        description=(
            "Play as the Human, on the right, against the Opponent. The up and down "
            "arrows move your Paddle, Space pauses and resumes, and Esc quits."
        ),
    )
    commands.add_parser(
        "measure-env",
        help="measure how many Steps a second the environment plays",
        description=(
            "Time one environment playing random Actions against the Opponent, "
            "drawing the Observation every Step, and print its Steps a second. "
            "Fails below the gate."
        ),
    )
    evaluate_command = commands.add_parser(
        "evaluate",
        help="evaluate a player in the Agent's place, on the regular set of seeds",
        description=(
            "Play an Evaluation of a player in the Agent's place: one Game against "
            "the Opponent for each of the regular set's 100 seeds. The player is a "
            "Scripted player, or a Milestone of a Run, after the Run, played "
            "greedily, on the CPU, by a network of the kind of the Run's player. "
            "Print its summary line. It never records anything in the Run, so it "
            "never changes the Milestone the Run chooses."
        ),
    )
    # Exactly one player to evaluate, of any kind: a Scripted player, or a learning
    # player's Milestone.
    player = evaluate_command.add_mutually_exclusive_group(required=True)
    player.add_argument(
        "--scripted",
        choices=list(SCRIPTED_PLAYERS),
        help="the Scripted player to evaluate",
    )
    player.add_argument(
        "--milestone",
        type=Path,
        metavar="MILESTONE_FILE",
        help="the Milestone to evaluate, a file in its Run's directory",
    )
    evaluate_command.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help=(
            "also add the Evaluation's line, as a Run's evaluations.jsonl keeps it, "
            "to the end of FILE"
        ),
    )
    commands.add_parser(
        "choose-width",
        help="choose the Opponent's offset width, by the Scripted players",
        description=(
            "Evaluate each Scripted player on the regular set against an Opponent "
            "of each offset width tried, print a table of their means, and choose "
            "the width by the rule of issue #7. Fails if no width passes."
        ),
    )
    train_command = commands.add_parser(
        "train",
        help="start a new Run of a learning player",
        description=(
            "Start a new Run of a learning player, from scratch: a Run directory, "
            "whose settings file records the settings and never changes, and "
            "training to the Budget. A player with Evaluations stops at its first "
            "Milestone good enough, and then gives its best Milestone its one "
            "Confirmation. It never starts from another Run's weights. It refuses "
            "the CPU unless --device cpu asks for it."
        ),
    )
    train_command.add_argument(
        "--player",
        required=True,
        choices=list(learning_players),
        help=(
            "the learning player to train: the Position learner, or a test "
            "environment's; the Agent comes later"
        ),
    )
    train_command.add_argument(
        "--name", required=True, help="the Run's short name, in its directory's name"
    )
    train_command.add_argument(
        "--seed", required=True, type=int, help="the Run's seed, below 1,000,000"
    )
    _add_device_option(train_command)
    _add_run_settings(train_command)
    resume_command = commands.add_parser(
        "resume",
        help="resume a Run that stopped, by its directory",
        description=(
            "Resume a Run that stopped, by Ctrl+C or by a crash, as the same Run: "
            "from its latest full checkpoint, with a new Game, to its end, and its "
            "Confirmation if it hasn't had it. A Run that had it never goes on. Its "
            "settings are read from its settings file and never change. It warns "
            "if the code is at another commit than the Run was created at, and "
            "refuses the CPU unless --device cpu asks for it."
        ),
    )
    resume_command.add_argument(
        "run", type=Path, metavar="RUN_DIRECTORY", help="the Run's directory"
    )
    _add_device_option(resume_command)
    args = parser.parse_args(argv)

    try:
        if args.command == "play":
            # Only the window needs pygame, so only the window's commands import
            # it.
            from pong_ai_bot.window import play_against_the_opponent

            play_against_the_opponent(seed=secrets.randbits(32))
        elif args.command == "measure-env":
            return _measure_env()
        elif args.command == "evaluate":
            return _evaluate(args, learning_players=learning_players, pong=pong)
        elif args.command == "choose-width":
            return _choose_width(pong=pong)
        elif args.command == "train":
            return _train(
                args,
                learning_players=learning_players,
                make_milestone_evaluation=make_milestone_evaluation,
            )
        elif args.command == "resume":
            return _resume(
                args,
                learning_players=learning_players,
                make_milestone_evaluation=make_milestone_evaluation,
            )
    except _Refused as refusal:
        print(refusal, file=sys.stderr)
        return 1
    return 0


class _Refused(Exception):
    """A command that refuses to do what it was asked, with the reason: `pong`
    prints it and exits with 1, having changed nothing."""


def _add_device_option(command: argparse.ArgumentParser) -> None:
    """The option of where `command`, a command that trains, trains."""
    command.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="where to train: the graphics card, the default, or the CPU",
    )


def _measure_env() -> int:
    from pong_ai_bot.env_speed import (
        MEASURED_STEPS,
        STEPS_PER_SECOND_GATE,
        measure_env_speed,
    )

    speed = measure_env_speed(PongEnv(), steps=MEASURED_STEPS, seed=0)
    print(
        f"{speed.steps:,} Steps in {speed.seconds:.2f} seconds: "
        f"{speed.steps_per_second:,.0f} Steps a second."
    )
    if speed.passes_the_gate:
        print(f"It passes the gate of {STEPS_PER_SECOND_GATE:,}.")
        return 0
    print(
        f"It is below the gate of {STEPS_PER_SECOND_GATE:,}. Profile it and speed up "
        "only the code, not the rules, the Observation or the sizes (issue #14)."
    )
    return 1


def _evaluate(
    args: argparse.Namespace,
    *,
    learning_players: Mapping[str, LearningPlayer],
    pong: PongMaker,
) -> int:
    """`pong evaluate`: the Evaluation, on the regular set, of a Scripted player,
    or of a Milestone after its Run, at the Milestone's Step."""
    if args.scripted is not None:
        evaluation, step = evaluate_scripted_player(args.scripted, pong=pong), None
    else:
        run, milestone = _milestone_to_evaluate(
            args.milestone, output=args.output, learning_players=learning_players
        )
        evaluation = evaluate_milestone(run, milestone, players=learning_players)
        step = milestone.step
    print(evaluation.summary_line(seed_set=SeedSet.REGULAR, step=step))
    if args.output is not None:
        line = evaluation.json_line(seed_set=SeedSet.REGULAR, step=step)
        with args.output.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    return 0


def _milestone_to_evaluate(
    path: Path,
    *,
    output: Path | None,
    learning_players: Mapping[str, LearningPlayer],
) -> tuple[RunDirectory, SavedFile]:
    """The Milestone at `path` and its Run, for `pong evaluate` to evaluate, adding
    its line to `output` if given. If it can't, it is _Refused."""
    not_a_milestone = _Refused(
        f"{path} is not a Milestone of a Run: a Run's directory keeps its "
        "Milestones, named by their Steps, such as milestone_000500000.pt."
    )
    if not (path.parent / SETTINGS_FILE).is_file():
        raise not_a_milestone
    run = _open_run(path.parent)
    milestone = next(
        (saved for saved in run.milestones() if saved.path.name == path.name), None
    )
    if milestone is None:
        raise not_a_milestone
    if not learning_players[run.player].evaluated:
        raise _Refused(
            f"The Run's player, {run.player}, has no Evaluation: a test "
            "environment's Games have no score to count them by."
        )
    if (
        output is not None
        and output.resolve() == (run.path / EVALUATIONS_FILE).resolve()
    ):
        raise _Refused(
            f"Only the Run itself adds to its {EVALUATIONS_FILE}, so that an "
            "Evaluation after it never changes the Milestone it chooses. Give "
            "--output another file."
        )
    return run, milestone


def _choose_width(*, pong: PongMaker) -> int:
    def evaluate_at(name: str, width: float) -> Evaluation:
        evaluation = evaluate_scripted_player(name, width=width, pong=pong)
        # Each Evaluation as it ends, since all of them take minutes.
        summary = evaluation.summary_line(seed_set=SeedSet.REGULAR)
        print(f"{name} at {width:.0%}: {summary}", flush=True)
        return evaluation

    choice = choose_width(evaluate_at)
    print()
    print(choice.report())
    return 0 if choice.chosen is not None else 1


def _add_run_settings(command: argparse.ArgumentParser) -> None:
    """An option for each of a Run's settings, named after it, which changes it
    from its default."""
    group = command.add_argument_group(
        "the Run's settings",
        "Each changes one of the Run's settings from its default. The settings "
        "file records them all, and they never change after.",
    )
    for field in fields(RunSettings):
        flag = "--" + field.name.replace("_", "-")
        default = field.default
        if field.type is bool:
            group.add_argument(
                flag,
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
                help=f"default {'on' if default else 'off'}",
            )
            continue
        kind = field.type
        assert isinstance(kind, type)
        if issubclass(kind, Enum):
            choices, metavar = list(kind), None
        else:
            choices, metavar = None, "N" if kind is int else "X"
        group.add_argument(
            flag,
            type=kind,
            choices=choices,
            default=argparse.SUPPRESS,
            metavar=metavar,
            help=f"default {_shown(default)}",
        )


def _shown(default: object) -> str:
    """A setting's default, as `pong train --help` shows it."""
    if isinstance(default, int):
        return f"{default:_}"
    return str(default)


_STOPPED_BY_CTRL_C = 130
"""`pong train`'s exit code when Ctrl+C stopped the Run, as is usual for a program
stopped by Ctrl+C: 128 and SIGINT's number, 2."""


def _training_device(name: str) -> torch.device:
    """The device `pong train` or `pong resume` trains on, by its `--device`
    `name`, printed. It refuses the CPU when PyTorch sees no graphics card,
    unless `name` asks for it (issue #11)."""
    import torch

    if name == "cuda" and not torch.cuda.is_available():
        raise _Refused(
            "PyTorch sees no graphics card, and on the CPU a Run trains about 20 "
            "times slower. Check that PyTorch is the cu126 build (README), or pass "
            "--device cpu to train on the CPU on purpose."
        )
    device = torch.device(name)
    if device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(device)})")
    else:
        print("Device: cpu")
    return device


def _train(
    args: argparse.Namespace,
    *,
    learning_players: Mapping[str, LearningPlayer],
    make_milestone_evaluation: MilestoneEvaluationMaker,
) -> int:
    device = _training_device(args.device)
    settings = RunSettings(
        **{
            field.name: getattr(args, field.name)
            for field in fields(RunSettings)
            if hasattr(args, field.name)
        }
    )
    run = RunDirectory.create(
        player=args.player, name=args.name, seed=args.seed, settings=settings
    )
    print(f"The Run's directory: {run.path}", flush=True)
    return _train_or_resume(
        run,
        device,
        resumed=False,
        player=learning_players[run.player],
        make_milestone_evaluation=make_milestone_evaluation,
    )


def _resume(
    args: argparse.Namespace,
    *,
    learning_players: Mapping[str, LearningPlayer],
    make_milestone_evaluation: MilestoneEvaluationMaker,
) -> int:
    device = _training_device(args.device)
    run = _open_run(args.run)
    # So that the Confirmation's set never becomes a set to choose by (issues #13,
    # #14).
    confirmation = run.confirmation()
    if confirmation is not None:
        raise _Refused(
            "The Run had its Confirmation, of the Milestone at Step "
            f"{confirmation.step:,}: a Run never goes on after it."
        )
    budget = run.settings.budget
    # The Run's end saves a Milestone at its Budget, which may come after its
    # latest checkpoint. A Run with Evaluations then evaluates it, and confirms
    # its best Milestone: until it has its Confirmation, its resume gives it.
    player = learning_players[run.player]
    evaluated = player.evaluated
    if not evaluated and any(saved.step == budget for saved in run.milestones()):
        print(
            f"The Run reached its Budget, {budget:,} Steps, already: there is "
            "nothing to resume."
        )
        return 0
    if not run.checkpoints():
        raise _Refused(
            "The Run has no full checkpoint to resume from: it saves its first at "
            f"Step {run.settings.checkpoint_every:,}, or when Ctrl+C stops it. "
            "Start a new Run instead."
        )
    return _train_or_resume(
        run,
        device,
        resumed=True,
        player=player,
        make_milestone_evaluation=make_milestone_evaluation,
    )


def _open_run(path: Path) -> RunDirectory:
    """The Run whose directory is `path`, with each warning of opening it, such as
    that the code is at another commit than the Run was created at, printed."""
    # Shown as a line of their own rather than as Python shows a warning, with the
    # line of code that gave it.
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        run = RunDirectory.open(path)
    for warning in warned:
        print(f"Warning: {warning.message}", file=sys.stderr)
    return run


def _train_or_resume(
    run: RunDirectory,
    device: torch.device,
    *,
    resumed: bool,
    player: LearningPlayer,
    make_milestone_evaluation: MilestoneEvaluationMaker,
) -> int:
    """Train the Run `run` on `device`, a new one or, if `resumed`, one resumed
    from its latest full checkpoint, with a new learner of its player, `player`,
    evaluated by the Evaluation `make_milestone_evaluation` makes if it has
    Evaluations, and say how it stopped."""
    # Only here: PyTorch takes seconds to import, and the other commands don't
    # need it.
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from pong_ai_bot.keep_awake import keep_windows_awake
    from pong_ai_bot.learner import Learner, LearnerSettings
    from pong_ai_bot.training import resume_run, train_run

    settings = run.settings
    # The network's first weights come from the Run's seed too. A resumed Run's
    # learner takes on its checkpoint's instead.
    torch.manual_seed(run.seed)
    network = player.make_network().to(device)
    learner = Learner(
        network,
        actions=player.actions,
        stack_length=player.stack_length,
        settings=LearnerSettings.of_run(settings),
        seed=run.seed,
    )
    # The Evaluation's network is its own, on the same device: a copy, which
    # leaves PyTorch's random generator as it was. The training is stopped while
    # it plays, its 100 Games together on the graphics card (#13).
    evaluation = (
        make_milestone_evaluation(copy.deepcopy(network), player.make_env)
        if player.evaluated
        else None
    )
    purge_step = None
    if resumed:
        step = run.checkpoints()[-1].step
        print(
            f"Resuming the Run at Step {step:,}, from its latest full checkpoint, "
            f"with a new Game. A new replay buffer fills for {settings.fill_steps:,} "
            "Steps before the learning goes on."
        )
        # TensorBoard hides every point from this Step on that was written before:
        # those written after the checkpoint, which the resumed Run writes again.
        purge_step = step + 1
    print(
        "Ctrl+C stops the Run once its Step is over, with a full checkpoint; a "
        "second Ctrl+C stops it at once, without one.",
        flush=True,
    )
    play = resume_run if resumed else train_run
    try:
        # The graphs go in the Run's directory, so that one TensorBoard on the
        # Runs' directory shows every Run. However the Run stops, the writer is
        # closed, with all it was given written.
        with SummaryWriter(log_dir=str(run.path), purge_step=purge_step) as writer:
            stopped = play(
                run,
                player.make_env(),
                learner,
                stay_awake=keep_windows_awake,
                writer=writer,
                evaluation=evaluation,
            )
    except KeyboardInterrupt:
        print("Stopped at once, with no checkpoint of the Step.", file=sys.stderr)
        return _STOPPED_BY_CTRL_C
    if stopped.by_ctrl_c:
        print(
            f"Ctrl+C stopped the Run at Step {stopped.step:,}, with a full checkpoint "
            "there."
        )
        return _STOPPED_BY_CTRL_C
    if stopped.step == settings.budget:
        print(f"The Run reached its Budget, {settings.budget:,} Steps.")
    else:
        print(
            f"The Run stopped at Step {stopped.step:,}, its first Milestone with a "
            f"mean of at least {settings.stop_mean:g} on the regular set."
        )
    return 0
