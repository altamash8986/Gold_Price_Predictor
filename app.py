import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# dataset
dataset = pd.read_csv("gold_price_prediction.csv")

# feature and target separate
x = dataset.drop("price", axis=1)
y = dataset[["price"]]

# split testing and training data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Standard Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# accuracy
model_accuracy = model.score(x_test_scaled, y_test)

# Prediction for visualization
y_pred = model.predict(x_test_scaled)
y_test_array = np.array(y_test).flatten()
y_pred_array = y_pred.flatten()

# Errors
mae_val = mean_absolute_error(y_test, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
r2_val = r2_score(y_test, y_pred)


# Plotting function
def create_scatter_plot(y_actual, y_pred):
    plt.scatter(y_actual, y_pred, c="green", s=40, label="Predictions")
    plt.plot(
        [min(y_actual), max(y_pred)],
        [min(y_actual), max(y_pred)],
        color="red",
        linestyle="--",
        label="Ideal Line",
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Gold Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Price_Plot.png")
    plt.close()


create_scatter_plot(y_test_array, y_pred_array)


# Prediction function
def gold_prediction(open_val, high_val, low_val, volume_val):
    input_data = pd.DataFrame(
        [[open_val, high_val, low_val, volume_val]],
        columns=["open", "high", "low", "volume"],
    )
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    return (
        f"{model_accuracy * 100:.2f} %",
        f"$ {round(float(prediction[0]), 4)}",
        round(mae_val, 2),
        round(rmse_val, 2),
        round(r2_val, 2),
        "Price_Plot.png",
    )


# GUI By Blocks
with gr.Blocks(title="Gold Price Prediction") as demo:
    gr.Markdown("# 💰 Gold Price Predictor")
    gr.Markdown(
        "This machine learning model predicts gold price based on Open, High, Low, and Volume.\n\n**Made by MOHD ALTAMASH**"
    )

    with gr.Row():
        open_input = gr.Slider(label="Open", minimum=1000, maximum=2100, value=1000)
        high_input = gr.Slider(label="High", minimum=1000, maximum=2100, value=1000)

    with gr.Row():
        low_input = gr.Slider(label="Low", minimum=1000, maximum=2100, value=1000)
        volume_input = gr.Slider(label="Volume", minimum=0, maximum=1000, value=0)

    predict_button = gr.Button("Click Here")
    gr.Markdown("## 📊 Prediction Output")

    with gr.Row():
        accuracy_output = gr.Label(label="Testing Accuracy",)
        price_output = gr.Label(label="Price")

    with gr.Row():
        mae_output = gr.Textbox(label="Mean Absolute Error")
        rmse_output = gr.Textbox(label="Root Mean Square Error")
        r2_output = gr.Textbox(label="R2 Score")

    plt_output = gr.Image(type="filepath", label="Actual vs Predicted Scatter Plot")

    predict_button.click(
        fn=gold_prediction,
        inputs=[open_input, high_input, low_input, volume_input],
        outputs=[
            accuracy_output,
            price_output,
            mae_output,
            rmse_output,
            r2_output,
            plt_output,
        ],
    )

demo.launch(share=True)

# print("---------------------------------------------------------------------------------------------------")

# GUI By Interface

# interface = gr.Interface(
#     fn=gold_prediction,
#     inputs=[
#         gr.Slider(label="Open", minimum=1000, maximum=2100, value=1000),
#         gr.Slider(label="High", minimum=1000, maximum=2100, value=1000),
#         gr.Slider(label="Low", minimum=1000, maximum=2100, value=1000),
#         gr.Slider(label="Volume", minimum=0, maximum=1000, value=0),
#     ],
#     outputs=[
#         gr.Textbox(label="Testing Accuracy", lines=1),
#         gr.Number(
#             label="Price",
#         ),
#         gr.Textbox(label="Mean Absolute Error"),
#         gr.Textbox(label="Root Mean Square Error"),
#         gr.Textbox(label="R2 Score"),
#         gr.Image(type="filepath", label="Actual vs Predicted Scatter Plot"),
#     ],
#     title="Gold Price Predictor",
#     description="This is a  Machine learning model predict the price of gold",
#     article="MADE BY MOHD ALTAMASH",
# )
#
# interface.launch(share=True)
