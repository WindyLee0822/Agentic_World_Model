"""Keep simulator infrastructure failures outside solver-visible cell output."""


class SimulatorObservationFailure(RuntimeError):
    """A candidate could not produce a simulator observation."""


def install_failure_boundary(world, predictor=None, predicted_error_type=()):
    # Install after prediction/setup wrappers. Their intentional API rejections
    # use AppWorld's plain Exception; infrastructure errors retain their types.
    request = world.requester.request
    execute = world.execute
    failure = None
    request_depth = 0

    if predictor is not None:
        predict = predictor._predict

        def guarded_predict(*args, **kwargs):
            # Verifier predictions run outside observation requests. Preserve
            # their original errors so the existing score-zero policy applies.
            if request_depth == 0:
                return predict(*args, **kwargs)
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
        nonlocal failure, request_depth
        if failure is not None:
            raise SimulatorObservationFailure("Simulator observation unavailable")
        request_depth += 1
        try:
            return request(*args, **kwargs)
        except Exception as exc:
            if type(exc) is not Exception:
                failure = exc
            raise
        finally:
            request_depth -= 1

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
