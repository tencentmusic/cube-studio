from myapp import app

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
