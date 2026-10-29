"""Keep simulator infrastructure failures outside solver-visible cell output."""


class SimulatorObservationFailure(RuntimeError):
    """A candidate could not produce a simulator observation."""


def install_failure_boundary(world, predictor=None, predicted_error_type=()):
    # Install after prediction/setup wrappers. Their intentional API rejections
    # use AppWorld's plain Exception; infrastructure errors retain their types.
    request = world.requester.request
    execute = world.execute
    failure = None

    if predictor is not None:
        predict = predictor._predict

        def guarded_predict(*args, **kwargs):
            try:
                return predict(*args, **kwargs)
            except predicted_error_type:
                raise
            except Exception as exc:
                # A plain SDK Exception must not be mistaken for a predicted
                # rejection. Errors repaired inside _predict never reach here.
                raise SimulatorObservationFailure(
                    "Simulator observation unavailable") from exc

        predictor._predict = guarded_predict

    def guarded_request(*args, **kwargs):
        nonlocal failure
        if failure is not None:
            raise SimulatorObservationFailure("Simulator observation unavailable")
        try:
            return request(*args, **kwargs)
        except Exception as exc:
            if type(exc) is not Exception:
                failure = exc
            raise

    def guarded_execute(code, *args, **kwargs):
        if failure is not None:
            raise SimulatorObservationFailure("Simulator observation unavailable") from failure
        try:
            output = execute(code, *args, **kwargs)
        finally:
            # AppWorld can stringify errors, and solver code can catch them.
            # Check host-side state before returning ANY output to the agent.
            if failure is not None:
                raise SimulatorObservationFailure("Simulator observation unavailable") from failure
        return output

    world.requester.request = guarded_request
    world.execute = guarded_execute
