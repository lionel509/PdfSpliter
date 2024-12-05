import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class TabularPreprocessor:
    def __init__(self, scale_features=True, impute_strategy="mean"):
        """
        Initializes the tabular data preprocessor.
        :param scale_features: Whether to scale numerical features.
        :param impute_strategy: Strategy for imputing missing values (e.g., 'mean', 'median').
        """
        self.scale_features = scale_features
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def process(self, data_path):
        """
        Processes a tabular dataset.
        :param data_path: Path to the CSV file.
        :return: Preprocessed dataframe as a numpy array.
        """
        # Load the dataset
        df = pd.read_csv(data_path)

        # Separate numerical and categorical features
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = df.select_dtypes(include=['object']).columns

        # Impute missing values
        df[numerical_features] = self.imputer.fit_transform(df[numerical_features])
        df[categorical_features] = df[categorical_features].fillna("Missing")

        # Scale numerical features
        if self.scale_features:
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])

        # One-hot encode categorical features
        if not categorical_features.empty:
            encoded_cats = pd.DataFrame(
                self.encoder.fit_transform(df[categorical_features]),
                columns=self.encoder.get_feature_names_out(categorical_features)
            )
            df = pd.concat([df[numerical_features], encoded_cats], axis=1)

        return df.values
