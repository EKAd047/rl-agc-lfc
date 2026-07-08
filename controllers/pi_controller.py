"""
Conventional PI-based AGC controller.
Used as the baseline to compare against the PPO agent.
"""


class PIController:
    """
    Simple PI controller on Area Control Error (ACE).
    For a single area (no tie-line), ACE = beta * delta_f.
    """

    def __init__(self, kp, ki, beta):
        self.kp = kp
        self.ki = ki
        self.beta = beta
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def compute(self, delta_f, dt):
        ace = self.beta * delta_f

        self.integral += ace * dt

        # Negative sign: positive ACE (frequency too high) should
        # reduce P_ref, negative ACE (frequency too low) should raise it.
        P_ref = -(self.kp * ace + self.ki * self.integral)

        return P_ref