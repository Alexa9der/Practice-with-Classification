from lib.imports import *

class DataEncoder:
    """
    A class for performing label encoding and one-hot encoding transformations on DataFrame columns.

    Attributes:
        _le_job (LabelEncoder): LabelEncoder instance for 'JobRole' column encoding.
        _le_business (LabelEncoder): LabelEncoder instance for 'BusinessTravel' column encoding.
        enc (OneHotEncoder): OneHotEncoder instance for one-hot encoding transformations.

    Methods:
        label_encoder(df: pd.DataFrame, what_data=None) -> pd.DataFrame:
            Applies LabelEncoder transformation to 'JobRole' and 'BusinessTravel' columns.

        one_hot_encoder(df: pd.DataFrame, mask_f_ohe: list, what_data=None) -> pd.DataFrame:
            Applies OneHotEncoder transformation to selected columns in the DataFrame.

    """
    def __init__(self):
        """
        Initialize the DataEncoder class by creating LabelEncoder instances for 'JobRole' and 'BusinessTravel'
        columns, and an instance of OneHotEncoder.
        """
        self._le_job = LabelEncoder()
        self._le_business = LabelEncoder()
        self.enc = OneHotEncoder(drop='first', sparse=False)

    def label_encoder(self, df: pd.DataFrame, what_data=None) -> pd.DataFrame:
        """
        Apply LabelEncoder transformation to 'JobRole' and 'BusinessTravel' columns in the DataFrame.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing the data to be encoded.
            what_data (str): Indicates whether to use 'fit_transform' if 'train' or 'transform' if None.

        Returns:
            pd.DataFrame: Transformed DataFrame after applying LabelEncoder.
        """
        if what_data == "train":
            le_jr = self._le_job.fit_transform(df['JobRole'])
            cat_bt = self._le_business.fit_transform(df['BusinessTravel'])
        else:
            le_jr = self._le_job.transform(df['JobRole'])
            cat_bt = self._le_business.transform(df['BusinessTravel'])

        df['JobRole'] = le_jr
        df['BusinessTravel'] = cat_bt
        
        return df
    
    def one_hot_encoder(self, df: pd.DataFrame, mask_f_ohe: list, what_data=None) -> pd.DataFrame:
        """
        Apply OneHotEncoder transformation to selected columns in the DataFrame.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing the data to be encoded.
            mask_f_ohe (list): List of column names to be encoded using OneHotEncoder.
            what_data (str): Indicates whether to use 'fit_transform' if 'train' or 'transform' if None.

        Returns:
            pd.DataFrame: Transformed DataFrame after applying OneHotEncoder.
        """
        if what_data == "train":
            data = self.enc.fit_transform(df[mask_f_ohe])
        else:
            data = self.enc.transform(df[mask_f_ohe])

        encoded_data = pd.DataFrame(data=data, columns=self.enc.get_feature_names_out(mask_f_ohe))
        df = df.drop(mask_f_ohe, axis=1)

        df = pd.concat([df.reset_index(drop=True), encoded_data.reset_index(drop=True)], axis=1)

        return df

class DataProcessor:
    """
    A class for processing data by dividing, calculating adequate values, and correcting columns.

    Attributes:
        data (pd.DataFrame): The input DataFrame to be processed.
        out_d (list): List of column names for adequate value calculation and correction.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataProcessor object.

        Parameters:
            data (pd.DataFrame): Input DataFrame to be processed.
        """
        self.data = data
        self.out_d = ["Age", "TotalWorkingYears", 'YearsAtCompany', 'YearsInCurrentRole',
                      'YearsSinceLastPromotion', 'YearsWithCurrManager', "DistanceFromHome"]

    def last_adequate_values(self, columns: list) -> dict:
        """
        Calculates the last adequate values for specified columns in the input DataFrame.

        Parameters:
            columns (list): List of column names for which last adequate values need to be calculated.

        Returns:
            dict: A dictionary containing column names as keys and value ranges as values.
                  The value range is represented as a list containing two elements: minimum and maximum last adequate values.
        """
        max_corect_data = {}
        for column in columns:
            max_d = 1
            min_d = 100
            yer = sorted(pd.unique(self.data[column]))

            for i in yer:
                if max_d < i and max_d + 100 > i:
                    max_d = i
                if min_d > i:
                    min_d = i

            max_corect_data[column] = [min_d, max_d]

        return max_corect_data

    def data_correction(self) -> pd.DataFrame:
        """
        Corrects specified columns in the input DataFrame based on calculated adequate values and logic.

        Returns:
            pd.DataFrame: DataFrame with corrected columns and unchanged non-corrected columns.
        """
        lav = self.last_adequate_values(columns=self.out_d)  # Calculate adequate values automatically

        df = self.data[[*lav.keys(), "EmployeeNumber"]].copy()
        for k, v in lav.items():
            if k == "Age":
                q = df[(df[k] > v[1])]
                df.iloc[q.index, list(lav).index(k)] = q["TotalWorkingYears"] + 18
            elif k == "TotalWorkingYears":
                q = df[df[k] > v[1]]
                df.iloc[q.index, list(lav).index(k)] = q["Age"] - 18
            elif k != "EmployeeNumber":
                q = df[df[k] > v[1]]
                norm_data = df[df[k] < v[1]]
                df.iloc[q.index, list(lav).index(k)] = np.random.randint(low=v[0], high=v[1], size=len(q))

        df_2 = self.data[[i for i in self.data.columns if i not in lav.keys()]]
        df = df.join(df_2.set_index("EmployeeNumber"), on="EmployeeNumber", how='left')

        return df

class Scaler:
    """
    A class for scaling features in a DataFrame using the StandardScaler.

    Attributes:
        scaler (StandardScaler): The StandardScaler instance used for scaling.

    Methods:
        fit_transform(data: pd.DataFrame) -> pd.DataFrame:
            Fits the scaler to the data and transforms it.

        transform(data: pd.DataFrame) -> pd.DataFrame:
            Transforms the data using the pre-fitted scaler.
    """
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the scaler to the data and transforms it.

        Parameters:
            data (pd.DataFrame): Input DataFrame containing features and possibly the 'EmployeeNumber' column.

        Returns:
            pd.DataFrame: Scaled DataFrame with the scaled features and the 'EmployeeNumber' column.
        """
        col_for_scalers = [i for i in data.columns if i != "EmployeeNumber"]
        s_train = self.scaler.fit_transform(data[col_for_scalers])
        scaled_data = data.copy()
        scaled_data[col_for_scalers] = s_train
        return scaled_data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data using the pre-fitted scaler.

        Parameters:
            data (pd.DataFrame): Input DataFrame containing features and possibly the 'EmployeeNumber' column.

        Returns:
            pd.DataFrame: Scaled DataFrame with the scaled features and the 'EmployeeNumber' column.
        """
        col_for_scalers = [i for i in data.columns if i != "EmployeeNumber"]
        s_train = self.scaler.transform(data[col_for_scalers])
        scaled_data = data.copy()
        scaled_data[col_for_scalers] = s_train
        return scaled_data

def divide_data_qual_quan(data: pd.DataFrame, connection_column: str = None,
                          num=15) -> tuple:
    """
    Divides data into qualitative (train_qual) and quantitative (train_quan) parts.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing the data to be divided.
        connection_column (str): Column name for connecting data between the two parts (default is None).
        num (int): Maximum number of unique values to consider a column as qualitative (default is 15).

    Returns:
        tuple: A tuple containing two DataFrames - train_qual (qualitative data) and train_quan (quantitative data).
    """
    nunique_ = data.nunique()
    mask_for_qua = list(nunique_[nunique_ <= num].index)

    train_qual = data.loc[:, mask_for_qua].copy()  # qualitative (make a copy)
    train_quan = data.loc[:, ~data.columns.isin(mask_for_qua)]  # quantitative

    if connection_column:
        train_qual.loc[:, connection_column] = data[connection_column]

    return train_qual, train_quan

def calculate_vif(data: pd.DataFrame, vif_threshold: float = 5.0) -> pd.DataFrame:
    """
    Calculate the Variance Inflation Factor (VIF) for each variable in the input DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing the variables.
        vif_threshold (float): The threshold value for VIF indicating multicollinearity. Default is 5.0.

    Returns:
        pd.DataFrame: A DataFrame containing the variable names and their corresponding VIF values.

    Example:
        vif_result = calculate_vif(train_data, vif_threshold=10.0)
        print(vif_result)
    """
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns

    # Calculate VIF values with a try-except block to handle potential division by zero
    vif_values = []
    for i in range(data.shape[1]):
        try:
            r_squared_i = sm.OLS(data.iloc[:, i], data.drop(data.columns[i], axis=1)).fit().rsquared
            vif = 1. / (1. - r_squared_i)
            vif_values.append(vif)
        except Exception as e:
            vif_values.append(float('inf'))  # Set to infinity if division by zero occurs
    vif_data["VIF"] = vif_values

    # Filter variables exceeding the VIF threshold
    vif_data = vif_data[vif_data["VIF"] > vif_threshold]
    
    return vif_data

def multicollinearity(corr: pd.core.frame.DataFrame, threshold: float = 0.5) -> dict:
    """
    Identify variables showing multicollinearity based on a given correlation matrix.

    Parameters:
        corr (pd.DataFrame): The correlation matrix containing correlations between variables.
        threshold (float, optional): The threshold value to consider correlations as significant for multicollinearity.
                                     Default is 0.5.

    Returns:
        dict: A dictionary where keys are variable names and values are lists of multicollinear variable names.
              The dictionary represents the variables that exhibit multicollinearity with each variable.

    Example:
        # Calculate correlation matrix
        correlation_matrix = data.corr()

        # Identify multicollinear variables
        multicollinear_vars = multicollinearity(correlation_matrix, threshold=0.6)
        print(multicollinear_vars)
    """
    multicol = {}
    for i in corr:
        multicol[i] = corr.loc[i, (corr[i] > threshold) | (corr[i] < -threshold)].index.tolist()
        if len(multicol[i]) == 1:
            del multicol[i]
    return multicol

def feature_importance_random_forest(X, y):
    """
    Evaluates feature importance using a Random Forest model.
    
    Parameters:
    X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
    y (numpy.ndarray): Target class vector of shape (n_samples,).
    
    Returns:
    feature_importances (numpy.ndarray): Array of feature importance scores.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    feature_importances = {col_name : round(np.abs(coef),3) for col_name, coef in 
                                 zip(X.columns, model.feature_importances_)}
#     model.feature_importances_
    return feature_importances

def logistic_regression_with_feature_selection(X, y, C=1.0):
    """
    Uses logistic regression with L1 regularization for binary classification with feature selection.
    
    Parameters:
    X (numpy.ndarray): Feature matrix (n_samples, n_features).
    y (numpy.ndarray): Target class vector (n_samples,).
    C (float): Inverse regularization strength (default: 1.0).
    
    Returns:
    selected_features (list): List of selected features.
    """
    # Create a logistic regression model with L1 regularization
    model = LogisticRegression(penalty='l1', C=C, solver='liblinear')
    
    # Fit the model
    model.fit(X, y)
    
    # Get absolute feature coefficients as feature importances
    feature_importances = {col_name : round(np.abs(coef),3) for col_name, coef in zip(X.columns, model.coef_[0])}
    
    return feature_importances

def rfe_feature_selection_with_cross_val(X, y, model=None, metric="f1"):
    """
    Performs feature selection using the Recursive Feature Elimination (RFE) method
    and cross-validation to choose the best set of features.

    Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        model (object, optional): Model object for feature importance estimation. If not specified,
            Logistic Regression will be used. (default: None)
        metric (str, optional): Selected metric for evaluating model quality. Supported metrics:
            "accuracy" - accuracy, "precision" - precision, "recall" - recall, "f1" - F1-score. (default: "f1")
    
    Returns:
        list: List of indices of the selected features.
    """
    if model is None:
        model = LogisticRegression(max_iter=1000)
    
    num_features = X.shape[1]
    best_score = 0.0
    best_selected_features = set()
    
    for num_selected_features in tqdm(range(1, num_features + 1)):
        rfe = RFE(model, n_features_to_select=num_selected_features)
        
        # Calculate cross-validated score using the selected metric
        scores = cross_val_score(rfe, X, y, cv=5, scoring=metric)
        current_score = np.mean(scores)
        
        if current_score > best_score:
            best_score = current_score
            best_selected_features = set(np.where(rfe.fit(X, y).support_)[0])
    
    return list(best_selected_features)

def pca_explanations(components, top_n=5):
    """
    Generate explanations for PCA components.

    This function generates explanations for each PCA component by identifying
    the top-n most important features for each component.

    Parameters:
    components (DataFrame): The DataFrame containing the PCA components.
    top_n (int, optional): The number of top features to consider for each component.
                           Defaults to 5.

    Returns:
    DataFrame: A DataFrame containing the top features and their importance
               for each PCA component.
    """
    print(f"Top-{top_n} most important features of each PCA component:")
    df = []
    for i in range(components.shape[0]):
        data = {}
        print(f"Component {i + 1}:")
        
        # Select the top_n most important features for the current component.
        top_features = components.loc[i].nlargest(top_n)
        
        for feature, importance in top_features.items():
            print(f"{feature}: {importance}")
            data[feature] = importance
        
        print("\n")

        df.append(data)
    
    # Create a DataFrame from the collected data and return it.
    return pd.DataFrame(df)

def svm_with_feature_selection(X, y):
    """
    Analyzes feature importance using coefficients from a linear SVM.

    Parameters:
    X (numpy.ndarray): Feature matrix (n_samples, n_features).
    y (numpy.ndarray): Target class vector (n_samples,).
    feature_names (list): List of feature names.

    Returns:
    None (Displays a bar plot of feature coefficients.)
    """
    # Create a linear SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X, y)
    
    coef_dict = {col_name : np.abs(coef) for col_name, coef in zip(X.columns, svm_model.coef_[0])}
    
    return coef_dict 
# model 
def select_model(X_train, y_train, models: type = None, gs_cv=5, cvs_cv = 5, scoring=None):
    """
    Selects the best weak classifier model based on cross-validation.
    
    Parameters:
        X_train (array-like): Matrix of training features.
        y_train (array-like): Vector of target variables for training.
        models (tuple): Tuple containing model name, model instance, and parameters for RandomizedSearchCV.
        cv (int): Number of cross-validation splits.
        scoring (str): Scoring metric used for cross-validation (default is None).
        
    Returns:
        tuple: A tuple of dictionaries containing information about cross-validation scores and best parameters for each model.
    """
    if not models:
        models = [
            ("LogisticRegression", LogisticRegression(solver='liblinear'), {"C": [0.1, 1, 10], 
                                                                            "penalty": ['l1', 'l2']}),
            ("KNeighborsClassifier", KNeighborsClassifier(), {"n_neighbors": [3, 5, 7], 
                                                              "p": [1, 2]}),
            ("GaussianNB", GaussianNB(), {}),
            ("DecisionTreeClassifier", DecisionTreeClassifier(), {"max_depth": [None, 10, 20]})
        ]

    best_scores = {}
    best_params = {}

    for name, model, params in tqdm(models, desc="Processing models"):
        grid_search = GridSearchCV(model, params, cv=gs_cv, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        scores = cross_val_score(best_model, X_train, y_train, cv=cvs_cv, scoring=scoring, n_jobs=-1)

        best_scores[name] = scores
        best_params[name] = grid_search.best_params_
        
    return best_scores, best_params


def build_estimator(base_model, n_estimators=50, learning_rate=1.0, algorithm='boost', problem_type='classification', **kwargs): 
    """
    Build an estimator with a chosen base model and algorithm.

    Parameters:
        base_model (object): Weak model object that will be used as the base.
        n_estimators (int): Number of base models in the estimator (default 50).
        learning_rate (float): Learning rate that controls the influence of each base model (default 1.0).
        algorithm (str): Algorithm to choose the estimator ('boost' for Boosting or 'bagging' for Bagging).
        problem_type (str): Type of problem ('classification' for classification problems or 'regression' for regression problems).
        **kwargs: Additional arguments that can be passed during estimator initialization.

    Returns:
        object: Estimator with the chosen base model and algorithm.
    """
    algorithms = {
        'boost': (AdaBoostClassifier, AdaBoostRegressor),
        'bagging': (BaggingClassifier, BaggingRegressor)
    }

    if algorithm not in algorithms:
        raise ValueError("Unknown algorithm. Enter 'boost' for Boosting or 'bagging' for Bagging.")

    classifier_cls, regressor_cls = algorithms[algorithm]

    if problem_type == 'classification':
        if algorithm == 'bagging':
            estimator = classifier_cls(base_estimator=base_model, n_estimators=n_estimators, **kwargs)
        else:
            estimator = classifier_cls(base_estimator=base_model, n_estimators=n_estimators, learning_rate=learning_rate, **kwargs)
    elif problem_type == 'regression':
        estimator = regressor_cls(base_estimator=base_model, n_estimators=n_estimators, learning_rate=learning_rate, **kwargs)
    else:
        raise ValueError("Unknown problem type. Enter 'classification' for classification or 'regression' for regression.")

    return estimator

# vizual 
def plot_feature_importance(feature_importances):
    """
    Visualizes feature importance using model coefficients and displays mean, median, and mode.

    Parameters:
    feature_importances (dict): Dictionary containing feature names as keys and their importances as values.
    """
    importance_values = list(feature_importances.values())
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    features = [feat for feat, _ in sorted_features]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(features)), importance_values)
    plt.xticks(range(len(features)), features, rotation=45, ha="right")
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')

    # Calculate and display mean, median, and mode
    mean_value = np.mean(importance_values)
    median_value = np.median(importance_values)

    plt.axhline(y=mean_value, color='r', linestyle='--', label='Mean')
    plt.axhline(y=median_value, color='g', linestyle='--', label='Median')

    plt.legend()
    plt.tight_layout()

    plt.show()

def plot_boxplots(data: dict) -> None:
    """
    Plot boxplots for the provided data dictionary.

    Parameters:
        data (dict): A dictionary containing data to plot, where keys are labels and values are lists of scores.

    Returns:
        None
    """
    fig, ax = plt.subplots()

    data_to_plot = []
    labels = []
    for k, v in data.items():
        data_to_plot.append(v)
        labels.append(k)
        print(f"{k:.^60}")
        print(f"mean: {np.mean(v)}")
        print(f"std: {np.std(v)}")
        print(f"median: {np.median(v)}")
        print(f"scor:\n{v},\n")
        
#         print(f"mean: {np.mean(v)}")
        
        
        
    ax.boxplot(data_to_plot, labels=labels)

    plt.xlabel('Labels')  
    plt.ylabel('Scores')  
    plt.title('Boxplot of Scores')
    ax.set_xticklabels(labels, rotation=45)
    plt.show()

# scor
def score(X: pd.DataFrame, y: np.array, models: list) -> None:
    """
    Evaluate the performance of classification models using cross-validation.

    Parameters:
        X (pd.DataFrame): Features matrix.
        y (np.array): Target vector.
        models (dict): Dictionary containing the classification models to evaluate.

    Returns:
        None
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model in models:
        print(model.__class__.__name__)
        y_pred = cross_val_predict(model, X, y, cv=skf)
        print(classification_report(y, y_pred))
        
        
# probability settings

def best_threshold(model, X_training, y_training, metric_function):
    """
    Find the best threshold for a given model based on a specified evaluation metric.

    Parameters:
        model (object): Model object to find the best threshold for.
        X_training (array-like): Training data features.
        y_training (array-like): Training data target.
        metric_function (function): Metric function used for evaluation.

    Returns:
        float: Best threshold found.
    """
    threshold = 0
    percent = 0.01
    best_metric_value = 0
    
    if model.__class__.__name__ != "ExtraTreesClassifier":
        prob_pred_training = model.predict_proba(X_training)[:, 1]
        
        for j in range(100):
            pred_p = np.where(prob_pred_training > percent, 1, 0)
            metric_value = metric_function(y_training, pred_p)
            
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                threshold = percent
            
            percent += 0.01
    else:
        threshold = None
        
    return threshold

def thresholds(ensemble_models, X, y, evaluation_metric):
    """
    Find the best thresholds for a list of ensemble models based on a specified evaluation metric.

    Parameters:
        ensemble_models (list): List of ensemble model objects.
        X (array-like): Data features.
        y (array-like): Data target.
        evaluation_metric (function): Metric function used for evaluation.

    Returns:
        list: List of best thresholds for each model.
    """
    models_threshold = []
    for model in tqdm(ensemble_models):
        threshold = best_threshold(model, X, y, evaluation_metric)
        models_threshold.append(threshold)
    return models_threshold

def threshold_cross_val_score(models, X, y, metric, threshold, cv=5, n_jobs=-1):
    """
    Perform cross-validation scoring for models using a specified threshold.

    Parameters:
        models (object or list): Model object or list of model objects.
        X (array-like): Data features.
        y (array-like): Data target.
        metric (function): Metric function used for evaluation.
        threshold (float or list): Threshold value or list of threshold values.
        cv (int): Number of cross-validation splits.
        n_jobs (int): Number of CPU cores to use for parallel processing.

    Returns:
        dict: Dictionary containing scores for each model.
    """
    def model_score(model, X, y, metric, threshold):
        # Cross-validation scoring for a single model
        scores = []
        for _ in tqdm(range(cv), desc=f"Model: {model.__class__.__name__}"):
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            model.fit(X_train, y_train)
            
            # Make predictions
            pred_prob = model.predict_proba(X_test)
            if threshold is not None:
                pred = np.where(pred_prob[:, 1] > threshold, 1, 0)
            else:
                pred = model.predict(X_test)
            
            scores.append(metric(y_test, pred))
        return scores
    
    all_scores = {}
    if isinstance(models, list):
        all_scores = Parallel(n_jobs=n_jobs)(
            delayed(model_score)(m, X, y, metric, th) for m, th in tqdm(zip(models, threshold))
        )
        all_scores = {f"{m.__class__.__name__}_{m.estimators_[0].__class__.__name__}": scores for m, scores in zip(models, all_scores)}
    else:
        all_scores = model_score(models, X, y, metric, threshold)
        all_scores = {models.__class__.__name__: all_scores}
    return all_scores

def predict_with_threshold(model, X, pred_threshold):
    """
    Make predictions using a threshold for binary classification.

    Parameters:
        model (object): Model object for prediction.
        X (array-like): Data features.
        pred_threshold (float): Threshold for prediction.

    Returns:
        array-like: Predicted binary labels.
    """
    if model.__class__.__name__ != "ExtraTreesClassifier":
        pred_prob = model.predict_proba(X)
        pred = np.where(pred_prob[:, 1] > pred_threshold, 1, 0)
    else:
        pred = model.predict(X)
    return pred
        
