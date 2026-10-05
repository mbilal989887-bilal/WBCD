from flask import Flask, render_template, request
import model

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def dashboard():

    context = {
        "features": model.X.columns,
        "result": None
    }

    if request.method == "POST":
        form_data = {
            f: float(request.form[f])
            for f in model.X.columns
        }

        result, prob, input_df = model.predict_from_input(form_data)

        context.update({
            "result": result,
            "probability": prob,
            "prob_plot": model.probability_plot(prob),
            "compare_plot": model.user_vs_mean_plot(input_df),
            "importance_plot": model.feature_importance_plot()
        })

    return render_template("dashboard.html", **context)

if __name__ == "__main__":
    app.run(debug=True)
