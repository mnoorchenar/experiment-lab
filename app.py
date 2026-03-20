from flask import Flask, redirect, url_for
from routes.power import power_bp
from routes.ab_test import ab_test_bp
from routes.doe import doe_bp
from routes.sequential import sequential_bp

app = Flask(__name__)

app.register_blueprint(power_bp)
app.register_blueprint(ab_test_bp)
app.register_blueprint(doe_bp)
app.register_blueprint(sequential_bp)


@app.route("/")
def index():
    return redirect(url_for("power.power_page"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
