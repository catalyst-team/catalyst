import time

from catalyst.rl2 import RLRunner


class OffpolicyRLRunner(RLRunner):

    def _fetch_initial_buffer(self):
        replay_buffer = self.state.replay_buffer
        min_num_transitions = self.experiment.min_num_transitions

        buffer_size = len(replay_buffer)
        while buffer_size < min_num_transitions:
            replay_buffer.recalculate_index()

            num_trajectories = replay_buffer.num_trajectories
            num_transitions = replay_buffer.num_transitions
            buffer_size = len(replay_buffer)

            metrics = [
                f"fps: {0:7.1f}",
                f"updates per sample: {0:7.1f}",
                f"trajectories: {num_trajectories:09d}",
                f"transitions: {num_transitions:09d}",
                f"buffer size: "
                f"{buffer_size:09d}/{min_num_transitions:09d}",
            ]
            metrics = " | ".join(metrics)
            print(f"--- {metrics}")

            time.sleep(1.0)

    def _prepare_for_stage(self, stage: str):
        self._fetch_initial_buffer()
