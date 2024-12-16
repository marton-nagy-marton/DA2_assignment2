import pandas as pd
import numpy as np
from plotnine import *
from mizani.formatters import percent_format
import copy
from sklearn.decomposition import PCA
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.metrics

def plot_histogram(df, col, bw, xlabel):

    """
    Creates a histogram of the specified column in a DataFrame using the `plotnine` library.
    The histogram displays the percentage distribution of the column's values, with the y-axis
    normalized to represent percentages.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be plotted.
    col : str
        The name of the column in the DataFrame to plot.
    bw : float
        The bin width for the histogram.
    xlabel : str
        The label for the x-axis of the histogram.

    Returns:
    --------
    plotnine.ggplot
        A `plotnine` ggplot object representing the histogram.
    """
    
    p = (
        ggplot(df)
        + aes(x = col)
        + geom_histogram(
            aes(y=after_stat("count / np.sum(count)")),
            fill = 'lightgrey', 
            color = 'black', 
            binwidth = bw
        )
        + scale_x_continuous(
            expand=(0.01, 0),
        )
        + scale_y_continuous(
            expand=(0, 0.001),
            labels=percent_format()
        )
        + labs(
            x = xlabel,
            y = 'Percentage'
        )
        + theme_light()
    )
    return p

def plot_lowess(df, x, y, xtitle=None, ytitle=None, title=None, span=0.75):
    
    """
    Creates a scatter plot with a locally weighted regression (LOWESS) smoothed curve using the `plotnine` library.
    The plot shows the relationship between two variables with a smooth trend line.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be plotted.
    x : str
        The name of the column in the DataFrame to use for the x-axis.
    y : str
        The name of the column in the DataFrame to use for the y-axis.
    xtitle : str, optional
        The label for the x-axis. If None, the column name `x` will be used. Default is None.
    ytitle : str, optional
        The label for the y-axis. If None, the column name `y` will be used. Default is None.
    title : str, optional
        The title of the plot. Default is None.
    span : float, optional
        The smoothing parameter for the LOWESS curve. Controls the amount of smoothing; 
        larger values result in smoother curves. Default is 0.75.

    Returns:
    --------
    plotnine.ggplot
        A `plotnine` ggplot object representing the scatter plot with the LOWESS smoothed curve.
    """

    p = (
        ggplot(df)
        + aes(x = x, y = y)
        + geom_point(color = 'grey', alpha = 0.1)
        + geom_smooth(method = 'loess', se = False, span = span)
        + theme_light()
        + scale_x_continuous(expand=(0.01, 0.01))
        + scale_y_continuous(expand=(0.01, 0.01))
        + labs(x = xtitle if xtitle is not None else x,
               y = ytitle if ytitle is not None else y,
               title = title,
              )
    )
    return p

def plot_regscatter(x, y, prediction, xtitle=None, ytitle=None, title=None):
    
    """
    Creates a scatter plot of observed data points and overlays a regression line using the `plotnine` library.

    Parameters:
    -----------
    x : array-like
        The values for the x-axis.
    y : array-like
        The observed values for the y-axis.
    prediction : array-like
        The predicted values to be plotted as a regression line.
    xtitle : str, optional
        The label for the x-axis. If None, the variable name `x` will be used. Default is None.
    ytitle : str, optional
        The label for the y-axis. If None, the variable name `y` will be used. Default is None.
    title : str, optional
        The title of the plot. Default is None.

    Returns:
    --------
    plotnine.ggplot
        A `plotnine` ggplot object representing the scatter plot with a regression line.
    """

    return (
        ggplot()
        + geom_point(aes(x=x, y=y),fill="grey",color='grey',alpha=0.1)
        + geom_line(aes(x=x, y=prediction), colour="black", size = 1)
        + labs(x = xtitle if xtitle is not None else x,
               y = ytitle if ytitle is not None else y,
               title = title,
              )
        #+ annotate("text", x=(np.min(x)+0.1)*1.5, y=np.min(prediction)-0.05, label=text, size=8, color = 'black')
        + theme_bw()
    )

def lspline(series, knots):
    
    """
    Creates a linear spline design matrix for the given data series with specified knot points.

    A linear spline is a piecewise linear function with breaks (knots) at specified points. 
    This function returns a design matrix where each column corresponds to one segment of the spline.

    Parameters:
    -----------
    series : pandas.Series
        The input data series for which the spline design matrix will be generated.
    knots : list or scalar
        The knot points where the piecewise linear segments change slope. If a scalar is provided,
        it will be converted into a single-element list.

    Returns:
    --------
    numpy.ndarray
        A 2D NumPy array representing the design matrix for the linear spline, where each column 
        corresponds to one segment of the spline.

    Notes:
    ------
    - The function applies the ceiling at each knot point and adjusts subsequent segments accordingly.
    - The final column of the design matrix contains the residuals (unadjusted values after applying all knots).
    """

    def knot_ceil(vector, knot):
        vector_copy = copy.deepcopy(vector)
        vector_copy[vector_copy > knot] = knot
        return vector_copy

    if type(knots) != list:
        knots = [knots]
    design_matrix = None
    vector = series.values

    for i in range(len(knots)):
        # print(i)
        # print(vector)
        if i == 0:
            column = knot_ceil(vector, knots[i])
        else:
            column = knot_ceil(vector, knots[i] - knots[i - 1])
        # print(column)
        if i == 0:
            design_matrix = column
        else:
            design_matrix = np.column_stack((design_matrix, column))
        # print(design_matrix)
        vector = vector - column
    design_matrix = np.column_stack((design_matrix, vector))
    # print(design_matrix)
    return design_matrix

def lqspline(series: pd.Series, knot: float) -> np.array:
    """
    Generate a design matrix with:
    - A linear segment for the range 0-knot
    - A quadratic value for data bove the knot
    - A linear value for data above the knot

    Parameters:
    series (pd.Series): The input pandas Series.
    knot (float): The threshold value for the transition.

    Returns:
    np.array: The combined design matrix.
    """
    #linear component for range 0-knot
    linear_below_knot = series.apply(lambda x: x if x <= knot else knot).values.reshape(-1, 1)

    #linear segment for values above the knot
    linear_above_knot = series.apply(lambda x: max(0, x - knot)).values.reshape(-1, 1)

    #quadratic component for range 0-knot
    quadratic_above_knot = series.apply(lambda x: (x-knot)**2 if x > knot else 0).values.reshape(-1, 1)

    #combine all components into a single design matrix
    design_matrix = np.hstack((linear_below_knot, linear_above_knot, quadratic_above_knot))
    return design_matrix

def pca_train_test(train, test, columns_to_use):
    
    """
    Applies Principal Component Analysis (PCA) to the training and test datasets, reducing the specified columns to a single principal component.

    Parameters:
    -----------
    train : pandas.DataFrame
        The training dataset containing the features to be transformed.
    test : pandas.DataFrame
        The test dataset containing the features to be transformed.
    columns_to_use : list
        A list of column names from the datasets to include in the PCA transformation.

    Returns:
    --------
    tuple (numpy.ndarray, numpy.ndarray)
        - The transformed training dataset as a NumPy array with a single principal component.
        - The transformed test dataset as a NumPy array with a single principal component.

    Notes:
    ------
    - PCA is fit on the training dataset and applied to both the training and test datasets.
    - Reduces the dimensionality of the specified columns to 1 component.
    """

    pca = PCA(n_components = 1)
    tr = pca.fit_transform(train[columns_to_use])
    ts = pca.transform(test[columns_to_use])
    return tr, ts

def assign_stars(pval):
    
    """
    Assigns significance stars based on p-value thresholds.

    Parameters:
    ----------
    pval : float
        The p-value to evaluate.

    Returns:
    -------
    str
        A string containing significance stars:
        - '***' if pval < 0.01
        - '**' if 0.01 <= pval < 0.05
        - '*' if 0.05 <= pval < 0.1
        - '' if pval >= 0.1
    """
    
    if pval < 0.01:
        return '***'
    elif pval < 0.05:
        return '**'
    elif pval < 0.1:
        return '*'
    else:
        return ''

def get_compressed_logit_summary(model, raw_var, raw_name):
    
    """
    Generates a compressed summary of marginal effects from a logistic regression model for a specified variable.
    
    Parameters:
    ----------
    model : statsmodels.discrete.discrete_model.BinaryResultsWrapper
        A fitted logistic regression model.
    raw_var : str
        The variable name (as present in the model) for which to compute the summary.
    raw_name : str
        A human-readable name for the variable to use in the output dictionary.
    
    Returns:
    -------
    dict
        A dictionary summarizing the marginal effects of the specified variable in the model. The keys represent 
        different setups (e.g., linear spline, quadratic spline, polynomial terms), and the values indicate whether 
        the setup is present and the associated significance level:
            - 'Yes' or 'No' indicates the presence of a setup.
            - Stars ('*', '**', '***') indicate the significance level, based on p-values.
    """

    raw_var_types  = []
    model_vars = list(model.params.index)
    for var in model_vars:
        if raw_var in var:
            raw_var_types.append(var)
    out = {}
    #simple linear set up
    if raw_var in raw_var_types:
        out[raw_name] = (f'{model.get_margeff().margeff[model_vars.index(raw_var)-1]:.3f}'
                         + assign_stars(model.get_margeff().pvalues[model_vars.index(raw_var)-1]) #the -1 offset is because we do not get a marginal effect for the intercept
                         + f' ({model.get_margeff().summary_frame().loc[raw_var]['Std. Err.']:.3f})'
                        )
    else:
        out[raw_name] = 'No'
    
    if 'romani' not in raw_var:
        #lspline set up
        if sum(['lspline(' + raw_var in var for var in raw_var_types]) > 0:
            stars = []
            for var in raw_var_types:
                if 'lspline(' + raw_var in var:
                    stars.append(len(assign_stars(model.get_margeff().pvalues[model_vars.index(var)-1])))
            out[raw_name + ' lin. spline'] = 'Yes' + '*'*max(stars)
        else:
            out[raw_name + ' lin. spline'] = 'No'
        #lqspline set up
        if sum(['lqspline(' + raw_var in var for var in raw_var_types]) > 0:
            stars = []
            for var in raw_var_types:
                if 'lqspline(' + raw_var in var:
                    stars.append(len(assign_stars(model.get_margeff().pvalues[model_vars.index(var)-1])))
            out[raw_name + ' lin. quad. spline'] = 'Yes' + '*'*max(stars)
        else:
            out[raw_name + ' lin. quad. spline'] = 'No'
        #poly set up
        if sum([raw_var + ' **' in var for var in raw_var_types]) > 0:
            stars = []
            for var in raw_var_types:
                if raw_var + ' **' in var or raw_var == var:
                    stars.append(len(assign_stars(model.get_margeff().pvalues[model_vars.index(var)-1])))
            out[raw_name] = 'No'
            out[raw_name + ' polynom.'] = 'Yes' + '*'*max(stars)
        else:
            out[raw_name + ' polynom.'] = 'No'
    
    return out

def get_compressed_ols_summary(model, raw_var, raw_name):
    
    """
    Generate a compressed summary of an OLS regression model's results for a specified variable.

    Parameters:
    -----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The OLS regression results object containing model parameters and p-values.
    raw_var : str
        The variable name in the model for which the summary is generated.
    raw_name : str
        A human-readable name for the variable to be used as keys in the output dictionary.

    Returns:
    --------
    dict
        A dictionary summarizing the presence and significance of the specified variable in various
        forms (simple linear, linear spline, linear-quadratic spline, polynomial, and categorical).
        The values indicate significance with asterisks (*) if applicable or "No" if the variable
        is not present in the respective form.

        Keys in the output dictionary:
        - `raw_name`: Coefficient with significance stars or "No" for simple linear models.
        - `raw_name + ' lin. spline'`: "Yes" with significance stars or "No" for linear spline models.
        - `raw_name + ' lin. quad. spline'`: "Yes" with significance stars or "No" for linear-quadratic spline models.
        - `raw_name + ' polynom.'`: "Yes" with significance stars or "No" for polynomial models.
        - `raw_name + ' (cat.)'`: "Yes" with significance stars or "No" for categorical models.

    Notes:
    ------
    - Significance stars are based on the p-values of the model parameters:
      More stars indicate higher significance.
    - If the variable is not found in any form, the corresponding value in the dictionary will be "No".
    - The function handles variables in raw, lspline, lqspline, polynomial, and categorical forms, with
      exceptions for variables containing 'romani'.
    """

    raw_var_types  = []
    model_vars = list(model.params.index)
    for var in model_vars:
        if raw_var in var:
            raw_var_types.append(var)
    out = {}
    #simple linear set up
    if raw_var in raw_var_types:
        out[raw_name] = (f'{model.params[raw_var]:.3f}'
                         + assign_stars(model.pvalues[raw_var])
                         + f' ({model.HC1_se[raw_var]:.3f})'
                        )
    else:
        out[raw_name] = 'No'
    
    if 'romani' not in raw_var:
        #lspline set up
        if sum(['lspline(' + raw_var in var for var in raw_var_types]) > 0:
            stars = []
            for var in raw_var_types:
                if 'lspline(' + raw_var in var:
                    stars.append(len(assign_stars(model.pvalues[model_vars.index(var)])))
            out[raw_name + ' lin. spline'] = 'Yes' + '*'*max(stars)
        else:
            out[raw_name + ' lin. spline'] = 'No'
        #lqspline set up
        if sum(['lqspline(' + raw_var in var for var in raw_var_types]) > 0:
            stars = []
            for var in raw_var_types:
                if 'lqspline(' + raw_var in var:
                    stars.append(len(assign_stars(model.pvalues[model_vars.index(var)])))
            out[raw_name + ' lin. quad. spline'] = 'Yes' + '*'*max(stars)
        else:
            out[raw_name + ' lin. quad. spline'] = 'No'
        #poly set up
        if sum([raw_var + ' **' in var for var in raw_var_types]) > 0:
            stars = []
            for var in raw_var_types:
                if raw_var + ' **' in var or raw_var == var:
                    stars.append(len(assign_stars(model.pvalues[model_vars.index(var)])))
            out[raw_name] = 'No'
            out[raw_name + ' polynom.'] = 'Yes' + '*'*max(stars)
        else:
            out[raw_name + ' polynom.'] = 'No'
        #categorical set up
        if sum(['C(' + raw_var in var for var in raw_var_types]) > 0:
            stars = []
            for var in raw_var_types:
                if 'C(' + raw_var in var:
                    stars.append(len(assign_stars(model.pvalues[model_vars.index(var)])))
            out[raw_name + ' (cat.)'] = 'Yes' + '*'*max(stars)
        else:
            out[raw_name+ ' (cat.)'] = 'No'
    return out

def compute_metrics(train, models):
    """
    Compute evaluation metrics for multiple logit models on a given dataset.

    Parameters:
    -----------
    train : pandas.DataFrame
        The dataset containing true labels ('has_comm_worker') and the predicted
        values for each model. Prediction columns should be named following 
        the pattern '{model_name}_pred'.
    models : dict
        A dictionary with model names as keys and their corresponding trained 
        model objects as values. These model names must match the prefixes in 
        the prediction columns of the `train` DataFrame.

    Returns:
    --------
    dict
        A dictionary with metric names as keys ('R-squared', 'Brier-score',
        'Pseudo R-squared', 'Log-loss') and lists of computed metric values as
        values. The metrics are calculated for each model in the `models` dictionary
        in the same order.

    Metrics Computed:
    -----------------
    - R-squared: Coefficient of determination between true and predicted values.
    - Brier-score: Mean squared error between true and predicted values.
    - Pseudo R-squared: McFadden's pseudo R-squared obtained from the model object.
    - Log-loss: Negative log-likelihood of the true labels given the predicted probabilities.

    Notes:
    ------
    - The `train` DataFrame must include prediction columns for all models
      in the `models` dictionary.
    - Each model object in `models` must have a `prsquared` attribute to calculate
      the pseudo R-squared.

    """
    metrics = {
        "R-squared": lambda y_true, y_pred: sklearn.metrics.r2_score(y_true, y_pred),
        "Brier-score": lambda y_true, y_pred: sklearn.metrics.mean_squared_error(y_true, y_pred),
        "Pseudo R-squared": lambda model: model.prsquared,
        "Log-loss": lambda y_true, y_pred: -1 * sklearn.metrics.log_loss(y_true, y_pred),
    }
    results = {metric: [] for metric in metrics}
    
    for name, model in models.items():
        y_true = train["has_comm_worker"]
        y_pred = train[f"{name}_pred"]

        results["R-squared"].append(metrics["R-squared"](y_true, y_pred))
        results["Brier-score"].append(metrics["Brier-score"](y_true, y_pred))
        try:
            results["Pseudo R-squared"].append(metrics["Pseudo R-squared"](model))
        except AttributeError: #this happens if we feed an LPM model to the function
            results["Pseudo R-squared"].append(np.nan)
        results["Log-loss"].append(metrics["Log-loss"](y_true, y_pred))

    return results

def get_confusion_table(ytrue, ypred, cutoff):
    """
    Compute a confusion table and display the percentage of misclassified observations.

    Parameters:
    -----------
    ytrue : pd.Series
        A pandas Series containing the true binary labels (0 or 1) for the observations.
    ypred : pd.Series
        A pandas Series containing the predicted probabilities or scores for the observations.
    cutoff : float
        A threshold value used to convert predicted probabilities into binary predictions.

    Returns:
    --------
    confusion_table : pd.DataFrame
        A pandas crosstab showing the confusion table, where:
        - Rows represent the actual classes (0 or 1).
        - Columns represent the predicted classes (0 or 1).

    Prints:
    -------
    str
        The name of the ypred Series and the percentage of misclassified observations 
        based on the cutoff value.
    """
    ypred_cat = ypred.apply(lambda x: 1 if x >= cutoff else 0)
    confusion_table = pd.crosstab(ytrue, ypred_cat)
    print(f'{ypred.name} confused values for {((confusion_table.iloc[0,1]+confusion_table.iloc[1,0])/len(ytrue)*100):.2f}% of observations.')
    return confusion_table