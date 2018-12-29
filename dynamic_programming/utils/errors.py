class InvalidMoveError(ValueError):
    """
    Error raised when agent tries to take unpredicted action.
    """
    pass


class InvalidStateError(ValueError):
    """
    Error raised when one tries to set agent to invalid state.
    """
    pass

