# ⚡ Smart Energy Meter — Energy Monitoring & Bill Prediction

A Streamlit-based smart energy monitoring application that analyzes electricity consumption, estimates monthly electricity bills, predicts the next month's bill, detects abnormal consumption, identifies high-consuming devices, and provides energy optimization suggestions.

## 🚀 Features

* 📊 **Energy Consumption Monitoring**

  * Analyze total electricity consumption from uploaded CSV data.
  * View monthly energy consumption and billing information.

* 💰 **Electricity Bill Estimation**

  * Calculate energy charges based on configurable cost per kWh.
  * Include fixed monthly charges based on sanctioned load.

* 📈 **Next-Month Bill Prediction**

  * Uses Linear Regression to predict upcoming monthly energy consumption and estimated electricity bill.

* 🔍 **Anomaly Detection**

  * Detect unusual electricity consumption using statistical Z-score analysis.
  * Identify devices contributing to abnormal consumption.

* 📦 **Device-Level Energy Analysis**

  * Calculate energy consumption for individual devices.
  * Visualize device-wise energy usage.

* 🔧 **Power Optimization**

  * Detect idle/wasteful devices.
  * Detect device overuse.
  * Identify devices operating during unusual hours.
  * Estimate potential savings from reducing energy waste.

* 🧠 **Smart Insights**

  * Identify peak consumption hours.
  * Calculate average daily consumption.
  * Identify the highest-consuming device.
  * Provide actionable energy-saving suggestions.

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Plotly**

## 📂 Project Structure

```text
Smart-Energy-Meter/
│
├── app.py
├── sample_data.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## 📋 Dataset Format

The application accepts CSV files containing:

```text
date_time
global_active_power
```

For device-level analysis and optimization features, the dataset can also contain device power columns ending with `_w`, for example:

```text
fridge_w
fan_w
lights_w
tv_w
computer_w
ac_w
```

A sample dataset is included in this repository as `sample_data.csv`.

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/Dhruvdinasi/Smart-Energy-Meter.git
```

Navigate to the project directory:

```bash
cd Smart-Energy-Meter
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 How to Use

1. Launch the Streamlit application.
2. Upload `sample_data.csv` or another compatible energy consumption CSV.
3. Configure the electricity tariff and sanctioned load from the sidebar.
4. Adjust optimization and anomaly detection settings if required.
5. Explore:

   * Energy summary
   * Monthly billing
   * Device consumption
   * Bill prediction
   * Anomaly detection
   * Device attribution
   * Power optimization
   * Smart insights

## 🔮 Future Improvements

* Real-time IoT smart meter integration
* More advanced time-series forecasting models
* Machine learning-based anomaly detection
* User authentication and personalized dashboards
* Cloud deployment
* Historical consumption comparison
* Automated energy-saving recommendations

## 👨‍💻 Author

**Dhruv Dinasi**

GitHub: [Dhruvdinasi](https://github.com/Dhruvdinasi)
