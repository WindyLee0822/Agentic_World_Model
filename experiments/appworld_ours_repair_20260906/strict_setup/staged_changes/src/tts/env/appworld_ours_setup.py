"""Ours setup prediction; install after install_wm_requester.

Only installed for Ours with predict_all_actions enabled.
The wrapped requester already prevents native candidate task writes and supplies
logical supervisor state. Only its setup responses become prediction evidence.
"""

import os
import json


def wrap_setup_predictions(world, predictor):
    from tts.env.appworld_ours_wm import AppWorldOursWMPredictor

    if not isinstance(predictor, AppWorldOursWMPredictor):
        raise TypeError("Setup prediction experiment is restricted to AppWorld Ours")
    if not predictor.program_only:
        raise ValueError("Setup prediction requires generated Python observations")
    if os.environ.get("TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS", "0").lower() not in (
        "0", "false", "no", "off", "",
    ):
        raise ValueError("Exact clean-read shortcut is incompatible with setup prediction")

    predictor.setup_predictions_enabled = True
    predictor.n_setup_predictions = 0
    previous = world.requester.request

    def predict_setup(app, api, data, response):
        # These are synthetic AppWorld credentials, required in the predicted
        # observation itself. Shared computation redaction remains unchanged.
        # Encoding also distinguishes a valid JSON null from missing evidence.
        encoded = json.dumps(response, ensure_ascii=True)
        generated = predictor.predict_read(
            app, api, data, encoded, {"type": "string"}, dirty=False,
        )
        if isinstance(generated, str):
            try:
                decoded = json.loads(generated)
            except (ValueError, TypeError):
                if not isinstance(response, str):
                    raise RuntimeError("Simulator generated invalid setup JSON") from None
                decoded = generated
        else:
            # A generated program may decode the supplied JSON before printing.
            # This is still its predicted result, never the original evidence.
            decoded = generated
        predictor.n_setup_predictions += 1
        predictor.note_read(app, api, data, decoded)
        return decoded

    def request(_app_name, _api_name, client=None, raise_on_failure=None, show=False, track=True, **data):
        app, api = _app_name, _api_name
        setup = app in {"supervisor", "api_docs"} or api == "login"
        try:
            response = previous(
                app, api, client=client, raise_on_failure=raise_on_failure,
                show=show, track=track, **data,
            )
        except Exception as exc:
            # AppWorld surfaces API errors as plain Exception. Infrastructure
            # failures and simulator timeouts must retain their failure semantics.
            if not setup or type(exc) is not Exception:
                raise
            generated_error = predict_setup(app, api, data, str(exc))
            if not isinstance(generated_error, str):
                raise RuntimeError("Simulator did not generate a setup error string") from None
            raise Exception(generated_error) from None
        if not setup:
            return response
        # Model failure propagates. Never expose the evidence as a fallback.
        return predict_setup(app, api, data, response)

    world.requester.request = request
    return previous
