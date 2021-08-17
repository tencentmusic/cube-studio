

from myapp import (
    app,
    appbuilder,
    cache,
    conf,
    db,
    event_logger,
    get_feature_flags,
    is_feature_enabled,
    results_backend,
    security_manager,
)


config = app.config


@app.route("/health")
def health():
    return "OK"


@app.route("/healthcheck")
def healthcheck():
    return "OK"


@app.route("/ping")
def ping():
    return "OK"


