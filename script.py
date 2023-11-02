import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import random


class File:
    def __init__(self, path: str):
        # Read data from Excel sheet
        self.data = pd.read_excel(path, header=0)
        print(self.data)

    def get_data(self) -> pd.DataFrame:
        return self.data

    def dropColumns(self, cols: list):
        # Handle missing values
        self.data.dropna(inplace=True)
        # drop columns
        self.data.drop(columns=cols, inplace=True)


class Model:
    # Define models and their hyperparameters for grid search
    models = {
        "Linear Regression": {"model": LinearRegression(), "params": {}},
        "Random Forest": {
            "model": RandomForestRegressor(),
            "params": {"n_estimators": [50, 100, 150]},
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(),
            "params": {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.1, 0.5],
            },
        },
        "Support Vector Machine": {
            "model": SVR(),
            "params": {"C": [1, 10, 100], "kernel": ["linear", "rbf"]},
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(),
            "params": {"max_depth": [None, 10, 20, 30]},
        },
    }

    label_encoder_crop = LabelEncoder()
    label_encoder_state = LabelEncoder()
    scaler = StandardScaler()
    model = None

    def __init__(self, data_path: str):
        file = File(data_path)
        file.dropColumns(["Year"])
        data = file.get_data()
        self.raw_data = data

        # Encode categorical variables
        data["Crop"] = self.label_encoder_crop.fit_transform(data["Crop"])
        data["State"] = self.label_encoder_state.fit_transform(data["State"])
        self.data = data

        # Split features and target variable
        self.X = data.drop(columns=["Output"])
        self.y = data["Output"]
        print(self.X)
        print(self.y)

    def _split_data(self):
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.1, random_state=42
        )
        # Feature scaling

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def get_model(self):
        # Find the best model using GridSearchCV
        best_model = None
        best_mse = float("inf")
        self._split_data()

        for model_name, config in self.models.items():
            grid_search = GridSearchCV(
                config["model"],
                config["params"],
                cv=5,
                scoring="neg_mean_squared_error",
            )
            grid_search.fit(self.X_train, self.y_train)

            # Get the best model and its MSE
            model = grid_search.best_estimator_
            mse = mean_squared_error(self.y_test, model.predict(self.X_test))

            # Print results
            print(f"{model_name}: Mean Squared Error: {mse}")

            # Update best model if this one is better
            if mse < best_mse:
                best_mse = mse
                best_model = model

        print(f"Best Model: {best_model}")

        self.model = best_model
        return best_model

    def get_mathematical_model(self):
        if isinstance(self.model, LinearRegression):
            coefficients = self.model.coef_
            intercept = self.model.intercept_
            equation = "Output = "
            for i in range(len(coefficients)):
                equation += f"({coefficients[i]} * X{i}) + "
            equation += str(intercept)
            print("")
            print("")
            print("Mathematical Model:", equation)
        else:
            print("not linear regression")

    def get_area(self, state: str, crop: str) -> float:
        filtered_df = self.raw_data[
            (self.raw_data["Crop"] == self.label_encoder_crop.transform([crop])[0])
            & (self.raw_data["State"] == self.label_encoder_state.transform([state])[0])
        ]

        if not filtered_df.empty:
            # Get the first row from the filtered DataFrame
            dfs = filtered_df.iloc[0]["Area Cultivated"]
            return float(dfs)

        return 0

    def predict_output(
        self,
        crop: str,
        state: str,
        relative_humidity: float = 50,
        min_tempeature: float = 28,
        max_tempeature: float = 30,
        sunshine: float = 1000,
        rain_fall: float = 2000,
    ):
        print(f"what is the output of {crop} in {state}???????????")
        area = self.get_area(state, crop)

        if not area:
            ValueError(f"either state {state} or crop {crop} doesn't exist")

        input_data = pd.DataFrame(
            {
                "Crop": [crop],
                "Relative Humidity": [relative_humidity],
                "Min Temperature": [min_tempeature],
                "Max Temperature": [max_tempeature],
                "Radiation": [sunshine],
                "Rain Fall": [rain_fall],
                "Area Cultivated": [area],
                "State": [state],
            }
        )
        # Encode categorical variables in input data
        input_data["Crop"] = self.label_encoder_crop.transform(input_data["Crop"])
        input_data["State"] = self.label_encoder_state.transform(input_data["State"])

        # Scale input data
        input_data = self.scaler.transform(input_data)

        # Predict output using the best model
        output = self.model.predict(input_data)
        print("INPUT????????????", input_data)
        print("OUTPUT????????????", output)
        return output


# Now you can use the best_model for predictions.

modelObject = Model("./dataset.xlsx")
model = modelObject.get_model()
states = ["Bayelsa", "Cross River", "Akwa Ibom", "Abia", "Imo", "Rivers"]

prediction_params = [
    {
        "crop": "Oil Palm",
        "min_tempeature": 22,
        "max_tempeature": 33,
        "sunshine": 2000,
        "rain_fall": 2000,
        "relative_humidity": 50,
    },
    {
        "crop": "Plantain",
        "min_tempeature": 22,
        "max_tempeature": 28,
        "sunshine": 2100,
        "rain_fall": 2000,
        "relative_humidity": 50,
    },
    {
        "crop": "Cassava",
        "min_tempeature": 25,
        "max_tempeature": 30,
        "sunshine": 2100,
        "rain_fall": 2000,
        "relative_humidity": 65,
    },
]


for obj in prediction_params:
    for state in states:
        modelObject.predict_output(
            obj["crop"],
            state,
            obj["relative_humidity"],
            obj["min_tempeature"],
            obj["max_tempeature"],
            obj["sunshine"],
            obj["rain_fall"],
        )


modelObject.get_mathematical_model()
