import numpy as np
import kaggle_environments as ke

from alpha_zero.alphazero import StateStack2

    
def test_error_plays_twice():
    # player 1 plays twice
    ss = StateStack2()
    ss.step(0, 0)
    try:
        ss.step(0, 0)
    except ValueError as ve:
        if "Next to play" not in str(ve):
            raise ve


def test_correct_player_sequence():
    # player 1 plays followed by 2, some checks
    ss = StateStack2()
    ss.step(0, 2)
    current_state = np.zeros((6, 7), dtype=np.uint8)
    current_state[-1, 2] = 1
    assert np.all(ss.get_current_state() == current_state), (
        f"get_current\n{ss.get_current_state()}\nshould_match\n{current_state}")
    assert np.sum(ss.stack) == 1 + ss.stack.shape[0] * ss.stack.shape[1]
    ss.step(1, 2)
    # 2 ones for positions, and player rolled
    assert np.all(np.logical_or(ss.stack == 1, ss.stack == 0))
    assert np.sum(ss.stack) == 2, f"2 != {np.sum(ss.stack)}"

    # test "done" works
    ss = StateStack2()
    for i in range(ss.inarow * 2):
        done = ss.step(i % 2, i % 2)
        # -1 for indexing, -1 because p1 reaches 4 first
        if i == ss.inarow * 2 - 1 - 1:
            assert done
            break
        else:
            assert not done, f"finished after {i}, {ss}"


def test_error_play_full_row():
    # row is filled and played into - raises error
    ss = StateStack2()
    try:
        for i in range(ss.inarow * 2):
            ss.step(i % 2, 0)
        raise ke.errors.FailedPrecondition(f"Unexpected succcess {i}")
    except ke.errors.FailedPrecondition as fp:
        if i != ss.num_rows + 1:
            print(f"Unexpected failure {i}")
            raise fp


def test_error_play_finished_game():
    # Player wins and move is played - raises error
    ss = StateStack2()
    try:
        for i in range(ss.inarow * 2):
            ss.step(i % 2, i % 2)
        raise ke.errors.FailedPrecondition(f"Unexpected success {i}")
    except ke.errors.FailedPrecondition as fp:
        if i != ss.num_rows + 1:
            print(f"Unexpected failure {i}")
            raise fp


def test_init_first_row():
    init_state = np.zeros((6, 7, 2 * 4 + 1), dtype=np.uint8)
    init_state[-1, 0, 0] = 1
    init_state[-2:, 1, 0] = 1
    init_state[:7, 3, :2] = 1

    expected = np.array([1, 2, 0, 6, 0, 0, 0])

    ss = StateStack2(6, 7, 4, 4, preload_state=init_state)
    ss.init_first_empty_row()

    assert np.all(ss.first_empty_row == expected), (
        f"{ss.first_empty_row} != {expected}\n\n{ss.get_current_state()}")


if __name__ == "__main__":
    test_error_plays_twice()
    test_correct_player_sequence()
    test_error_play_full_row()
    test_error_play_finished_game()
    test_init_first_row()
