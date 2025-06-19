# Time-series-Forecasting
 Here's a README file description for your GitHub time series forecasting project, complete with emojis!

---

# 📈 Stock Price Prediction using Recurrent Neural Networks (RNNs) 🚀

## ✨ Project Overview

This project implements various Recurrent Neural Network (RNN) architectures – **Simple RNN**, **Long Short-Term Memory (LSTM)**, and **Gated Recurrent Unit (GRU)** – to forecast stock prices. We leverage historical stock data to train these models, aiming to predict future opening prices with a focus on demonstrating the capabilities of different sequential models in time series analysis.

---

## 📊 Data

The project uses historical stock price data for **GOOGL (Alphabet Inc.)** from April 1, 2020, to April 1, 2023. The data is downloaded directly using the `yfinance` library. We specifically focus on predicting the **'Open'** price.

---

## 🚀 Features

* **Data Acquisition**: Seamlessly download historical stock data using `yfinance`. 📥
* **Data Preprocessing**:
    * Splitting data into training and testing sets (80/20 split).
    * Min-Max Scaling to normalize data between 0 and 1 for optimal model performance.
    * Creation of sequences (look-back period of 50 time steps) for training and testing RNN models.
* **Model Implementation**:
    * **Simple RNN**: A basic RNN model to establish a baseline for sequential prediction.
    * **LSTM (Long Short-Term Memory)**: A more advanced RNN architecture designed to overcome the vanishing gradient problem, making it suitable for longer sequences.
    * **GRU (Gated Recurrent Unit)**: Another powerful RNN variant that simplifies the internal structure of LSTMs while retaining much of their performance.
* **Model Training**: Models are trained using `mean_squared_error` as the loss function, with appropriate optimizers (SGD for Simple RNN, Adam for LSTM/GRU).
* **Performance Evaluation**: Visual comparison of predicted prices against actual prices for each model.

---

## 🛠️ Tools & Technologies

* **Python** 🐍
* **Libraries**:
    * `numpy`: For numerical operations.
    * `pandas`: For data manipulation and analysis.
    * `yfinance`: To fetch historical stock data.
    * `datetime`: For handling dates.
    * `matplotlib`: For data visualization. 📊
    * `math`: For mathematical operations.
    * `scikit-learn`: For `MinMaxScaler` and evaluation metrics.
    * `tensorflow` / `keras`: For building and training RNN, LSTM, and GRU models.

---

## 📈 Results & Visualizations

The project provides visual comparisons of the predicted stock prices by each model against the actual stock prices. This allows for an intuitive understanding of how well each RNN architecture performs in forecasting.

*(Note: The provided code snippet only shows plots for Simple RNN and LSTM. The GRU plot would be similar if `regressorGRU` was defined and used.)*

---

## 🧠 Model Architectures

### Simple RNN

A sequential model with multiple `SimpleRNN` layers and `Dropout` for regularization, followed by a `Dense` output layer. Optimized with `SGD`.

```python
regressor = Sequential()
regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
regressor.add(SimpleRNN(units=50))
regressor.add(Dense(units=1, activation='sigmoid'))
regressor.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True), loss="mean_squared_error")
regressor.fit(X_train, y_train, epochs=20, batch_size=2)
```

### LSTM

A sequential model with two `LSTM` layers followed by `Dense` layers. Optimized with `Adam`.

```python
regressorLSTM = Sequential()
regressorLSTM.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressorLSTM.add(LSTM(50, return_sequences=False))
regressorLSTM.add(Dense(25))
regressorLSTM.add(Dense(1))
regressorLSTM.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
regressorLSTM.fit(X_train, y_train, batch_size=1, epochs=12)
```

*(Note: The GRU model structure would be similar to LSTM, replacing `LSTM` layers with `GRU` layers.)*

---

## 🏃 Getting Started

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Install dependencies**:
    ```bash
    pip install numpy pandas yfinance datetime matplotlib scikit-learn tensorflow keras
    ```
3.  **Run the script**:
    Execute the provided Python script to download data, train models, and generate plots.

---

## 🤝 Contribution

Contributions are welcome! If you have suggestions for improving the models, adding new features, or enhancing the documentation, please open an issue or submit a pull request.
