"""TODO: replace with a runnable starter artifact."""


class Policy:
    def reset(self, seed: int = 0) -> None:
        del seed

    def act(self, observation):
        del observation
        raise NotImplementedError("Implement the starter policy.")

