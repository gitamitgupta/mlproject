from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
	return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
	form = request.form

	data = CustomData(
		gender=form.get("gender", "male"),
		race_ethnicity=form.get("race_ethnicity", "group C"),
		parental_level_of_education=form.get("parental_level_of_education", "some college"),
		lunch=form.get("lunch", "standard"),
		test_preparation_course=form.get("test_preparation_course", "none"),
		reading_score=float(form.get("reading_score", 0)),
		writing_score=float(form.get("writing_score", 0)),
	)

	pred_df = data.get_data_as_data_frame()
	pipeline = PredictPipeline()
	prediction = pipeline.predict(pred_df)

	return render_template("result.html", prediction_text=f"Predicted Math Score: {prediction[0]:.2f}")


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)
