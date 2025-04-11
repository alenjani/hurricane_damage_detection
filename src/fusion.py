"""
Information fusion algorithm for combining multiple image predictions.

This module implements the information fusion approach described in the paper:
"Towards fully automated post-event data collection and analysis: pre-event and post-event information fusion"
"""

import numpy as np
import itertools as it
from scipy.special import logsumexp
import logging

logger = logging.getLogger(__name__)


class InformationFusion:
    """
    Information fusion algorithm to combine predictions from multiple images.

    This class implements a Bayesian approach to fuse information from multiple
    images of the same building to make a robust decision on damage classification.

    Parameters
    ----------
    f : callable
        A function that maps a binary vector to a probability value.
        For example, lambda alpha: np.ceil(np.mean(alpha)) returns 1 if at least
        half of the elements are 1, and 0 otherwise.
    """

    def __init__(self, f):
        """
        Initialize the information fusion algorithm.

        Parameters
        ----------
        f : callable
            A function that maps a binary vector to a probability value.
        """
        self.f = f

    def __call__(self, probabilities):
        """
        Apply the information fusion algorithm to combine multiple predictions.

        Parameters
        ----------
        probabilities : list or array-like
            A list of probabilities from individual image predictions.

        Returns
        -------
        float
            The fused probability representing the combined prediction.
        """
        if not probabilities:
            logger.warning("No probabilities provided for fusion. Returning 0.5.")
            return 0.5

        m = len(probabilities)
        log_probs = np.zeros(2 ** m)

        for i, alpha in enumerate(it.product((0, 1), repeat=m)):
            p_c_alpha = self.f(alpha)

            # Calculate the log probability
            log_p = np.log(p_c_alpha) if p_c_alpha > 0 else -np.inf

            for j in range(m):
                p = probabilities[j]
                # Avoid log(0) errors
                if alpha[j] == 1:
                    log_p += np.log(p) if p > 0 else -np.inf
                else:
                    log_p += np.log(1.0 - p) if p < 1 else -np.inf

            log_probs[i] = log_p

        # Use logsumexp for numerical stability
        result = np.exp(logsumexp(log_probs))

        # Ensure the result is between 0 and 1
        result = np.clip(result, 0, 1)

        return result


def fuse_building_predictions(image_predictions, fusion_method="mean"):
    """
    Fuse predictions from multiple images of the same building.

    Parameters
    ----------
    image_predictions : list
        List of probability values (between 0 and 1) from different images.
    fusion_method : str, optional
        Method to use for fusion: "mean", "max", "bayesian".
        Default is "mean".

    Returns
    -------
    float
        The fused probability value.
    """
    if not image_predictions:
        logger.warning("No predictions to fuse. Returning 0.5.")
        return 0.5

    if fusion_method == "mean":
        return np.mean(image_predictions)
    elif fusion_method == "max":
        return np.max(image_predictions)
    elif fusion_method == "bayesian":
        fuser = InformationFusion(lambda alpha: np.ceil(np.mean(alpha)))
        return fuser(image_predictions)
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")


def assign_damage_label(probability, thresholds=(0.4, 0.6)):
    """
    Assign a damage label based on the fused probability.

    Parameters
    ----------
    probability : float
        Probability of major damage (between 0 and 1).
    thresholds : tuple, optional
        (lower, upper) thresholds for decision making.
        If probability < lower, classify as non-major damage.
        If probability > upper, classify as major damage.
        Otherwise, return "no_decision".
        Default is (0.4, 0.6).

    Returns
    -------
    str
        The assigned label: "major_damage", "non_major_damage", or "no_decision".
    """
    lower, upper = thresholds

    if probability > upper:
        return "major_damage"
    elif probability < lower:
        return "non_major_damage"
    else:
        return "no_decision"


def decision_cost(decision, true_state, cost_matrix=None):
    """
    Calculate the cost of a decision given the true state.

    Parameters
    ----------
    decision : str
        The decision made: "major_damage", "non_major_damage", or "no_decision".
    true_state : str
        The true state: "major_damage" or "non_major_damage".
    cost_matrix : dict, optional
        A nested dictionary specifying the cost of each decision for each true state.
        Default is None, which uses a predefined cost matrix.

    Returns
    -------
    float
        The cost of the decision.
    """
    if cost_matrix is None:
        cost_matrix = {
            "major_damage": {
                "major_damage": 0,
                "non_major_damage": 1,
                "no_decision": 0.3
            },
            "non_major_damage": {
                "major_damage": 1,
                "non_major_damage": 0,
                "no_decision": 0.3
            }
        }

    return cost_matrix[true_state][decision]


def optimal_decision(probability, cost_matrix=None):
    """
    Make the optimal decision based on probability and cost matrix.

    Parameters
    ----------
    probability : float
        Probability of major damage (between 0 and 1).
    cost_matrix : dict, optional
        A nested dictionary specifying the cost of each decision for each true state.
        Default is None, which uses a predefined cost matrix.

    Returns
    -------
    str
        The optimal decision: "major_damage", "non_major_damage", or "no_decision".
    """
    if cost_matrix is None:
        cost_matrix = {
            "major_damage": {
                "major_damage": 0,
                "non_major_damage": 1,
                "no_decision": 0.3
            },
            "non_major_damage": {
                "major_damage": 1,
                "non_major_damage": 0,
                "no_decision": 0.3
            }
        }

    decisions = ["major_damage", "non_major_damage", "no_decision"]
    expected_costs = []

    for decision in decisions:
        # Calculate expected cost for this decision
        # Expected cost = P(major_damage) * cost(decision|major_damage) +
        #                P(non_major_damage) * cost(decision|non_major_damage)
        expected_cost = (
                probability * cost_matrix["major_damage"][decision] +
                (1 - probability) * cost_matrix["non_major_damage"][decision]
        )
        expected_costs.append(expected_cost)

    # Return the decision with minimum expected cost
    return decisions[np.argmin(expected_costs)]