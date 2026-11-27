"""A Run's settings, and their defaults, as the spec's table sets them (#17)."""

from pong_ai_bot.run_settings import Loss, RunSettings


def test_the_defaults_are_the_spec_s_table() -> None:
    settings = RunSettings()

    assert settings.budget == 10_000_000
    assert settings.replay_buffer_size == 1_000_000
    assert settings.fill_steps == 50_000
    assert settings.epsilon_start == 1.0
    assert settings.epsilon_end == 0.01
    assert settings.epsilon_decay_steps == 1_000_000
    assert settings.learning_rate == 1e-4
    # PyTorch's own default eps for Adam.
    assert settings.adam_eps == 1e-8
    assert settings.batch_size == 32
    assert settings.update_every == 4
    assert settings.target_copy_every == 1_000
    assert settings.loss is Loss.MSE
    assert settings.max_gradient_norm is None
    assert settings.discount == 0.99
    assert settings.double_dqn is True
    assert settings.checkpoint_every == 50_000
    assert settings.checkpoints_kept == 2
    assert settings.milestone_every == 500_000
    assert settings.stop_mean == 19.0
    assert settings.confirmation_gate == 18.0
