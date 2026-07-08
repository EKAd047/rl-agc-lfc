"""
Conventional PI-based AGC controller.
Used as the baseline to compare against the PPO agent.
"""


class PIController:
    """Simple PI controller on Area Control Error (ACE)."""

    def __init__(self, kp=0.0, ki=0.0):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def compute(self, ace, dt):
        # TODO: implement PI control law
        pass
