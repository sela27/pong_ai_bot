"""The window: the Observation as it is, scaled up, with a score bar outside it.

One pygame-ce window for every mode (issue #5). It shows the image draw_observation
draws, the one the Agent sees, scaled up without smoothing and with nothing drawn on
it, under a bar with both sides' Points. There is no separate drawing for people, so
a bug in the Agent's image shows to the eye.

The window isn't built test-first: it is checked by eye, with
docs/checklists/window.md.
"""

import time

import numpy as np
import numpy.typing as npt
import pygame

from pong_ai_bot.game import Score
from pong_ai_bot.observation import OBSERVATION_SIZE, draw_observation
from pong_ai_bot.play import HumanVsOpponent

FRAMES_PER_SECOND = 60
"""The Frames played each second in the modes with a Human, always (issue #5)."""

_SCALE = 8
"""How many screen pixels across each pixel of the Observation takes."""
_IMAGE_SIDE = OBSERVATION_SIZE * _SCALE
_BAR_HEIGHT = 56
_BAR_COLOR = (64, 64, 64)
_TEXT_COLOR = (255, 255, 255)
_TEXT_MARGIN = 20
_FONT_SIZE = 40


class Window:
    """The window, with the score bar on top and the scaled Observation under it."""

    def __init__(self, title: str, *, left: str, right: str) -> None:
        """Open the window. `left` and `right` name the players on each side."""
        pygame.init()
        self._screen = pygame.display.set_mode((_IMAGE_SIDE, _BAR_HEIGHT + _IMAGE_SIDE))
        pygame.display.set_caption(title)
        self._font = pygame.font.Font(None, _FONT_SIZE)
        self._next_frame = time.perf_counter()
        self._left = left
        self._right = right

    def show(
        self, observation: npt.NDArray[np.uint8], score: Score, note: str = ""
    ) -> None:
        """Show an Observation and the score, with a short note in the bar's middle."""
        self._screen.fill(_BAR_COLOR, pygame.Rect(0, 0, _IMAGE_SIDE, _BAR_HEIGHT))
        self._write(f"{self._left}  {score.left}", left=_TEXT_MARGIN)
        self._write(f"{score.right}  {self._right}", right=_IMAGE_SIDE - _TEXT_MARGIN)
        if note:
            self._write(note, centerx=_IMAGE_SIDE // 2)

        # Each row of the Observation is a row of the image, from the top wall down.
        rgb = np.repeat(observation[:, :, np.newaxis], 3, axis=2)
        image = pygame.image.frombuffer(
            rgb.tobytes(), (OBSERVATION_SIZE, OBSERVATION_SIZE), "RGB"
        )
        # Scaling without smoothing keeps every pixel a sharp square.
        scaled = pygame.transform.scale(image, (_IMAGE_SIDE, _IMAGE_SIDE))
        self._screen.blit(scaled, (0, _BAR_HEIGHT))
        pygame.display.flip()

    def wait_for_next_frame(self) -> None:
        """Wait until it's time for the next Frame, at FRAMES_PER_SECOND.

        A Frame that came late doesn't make the next ones hurry to catch up.
        """
        # pygame's Clock waits whole milliseconds, 16 for a 60th of a second, so it
        # runs at 62.5 Frames a second. Waiting for each Frame's own deadline keeps
        # the rate at 60.
        self._next_frame += 1 / FRAMES_PER_SECOND
        delay = self._next_frame - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_frame = time.perf_counter()

    def close(self) -> None:
        pygame.quit()

    def _write(self, text: str, **place: int) -> None:
        rendered = self._font.render(text, True, _TEXT_COLOR)
        rect = rendered.get_rect(centery=_BAR_HEIGHT // 2, **place)
        self._screen.blit(rendered, rect)


def play_against_the_opponent(seed: int) -> None:
    """Human vs the Opponent, until the window is closed or Esc is pressed.

    The Human holds the up and down arrows to move, and Space pauses and resumes.
    When the Game ends, the window stays on its last Frame.
    """
    play = HumanVsOpponent(seed=seed)
    window = Window("pong play", left="Opponent", right="Human")
    paused = over = False
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
            if not paused and not over:
                # The keys held now move the Paddle in this very Frame.
                keys = pygame.key.get_pressed()
                result = play.play_frame(up=keys[pygame.K_UP], down=keys[pygame.K_DOWN])
                over = result.terminated or result.truncated
            note = "Game over" if over else "Paused" if paused else ""
            window.show(draw_observation(play.state), play.state.score, note)
            window.wait_for_next_frame()
    finally:
        window.close()
