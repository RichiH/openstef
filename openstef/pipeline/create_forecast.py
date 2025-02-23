# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import pandas as pd
import structlog

from openstef.data_classes.model_specifications import ModelSpecificationDataClass
from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.feature_engineering.feature_applicator import (
    OperationalPredictFeatureApplicator,
)
from openstef.model.confidence_interval_applicator import ConfidenceIntervalApplicator
from openstef.model.fallback import generate_fallback
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.model.serializer import MLflowSerializer
from openstef.pipeline.utils import generate_forecast_datetime_range
from openstef.postprocessing.postprocessing import (
    add_prediction_job_properties_to_forecast,
)
from openstef.validation import validation


def create_forecast_pipeline(
    pj: PredictionJobDataClass,
    input_data: pd.DataFrame,
    mlflow_tracking_uri: str,
) -> pd.DataFrame:
    """Create forecast pipeline

    This is the top-level pipeline which included loading the most recent model for
    the given prediction job.

    Expected prediction job keys: "id",

    Args:
        pj (PredictionJobDataClass): Prediction job
        input_data (pd.DataFrame): Training input data (without features)
        mlflow_tracking_uri (str): MlFlow tracking URI

    Returns:
        pd.DataFrame with the forecast

    """
    # Load most recent model for the given pid
    model, model_specs = MLflowSerializer(
        mlflow_tracking_uri=mlflow_tracking_uri
    ).load_model(experiment_name=str(pj["id"]))
    return create_forecast_pipeline_core(pj, input_data, model, model_specs)


def create_forecast_pipeline_core(
    pj: PredictionJobDataClass,
    input_data: pd.DataFrame,
    model: OpenstfRegressor,
    model_specs: ModelSpecificationDataClass,
) -> pd.DataFrame:
    """Create forecast pipeline (core)

    Computes the forecasts and confidence intervals given a prediction job and input data.
    This pipeline has no database or persisitent storage dependencies.

    Expected prediction job keys: "resolution_minutes", "horizon_minutes", "id", "type",
        "name", "quantiles"

    Args:
        pj (PredictionJobDataClass): Prediction job.
        input_data (pandas.DataFrame): Input data for the prediction.
        model (OpenstfRegressor): Model to use for this prediction.
        model_specs (ModelSpecificationDataClass): Model specifications.

    Returns:
        forecast (pandas.DataFrame)
    """
    logger = structlog.get_logger(__name__)

    fallback_strategy = "extreme_day"  # this can later be expanded

    # Validate and clean data
    validated_data = validation.validate(pj["id"], input_data, pj["flatliner_treshold"])

    # Add features
    data_with_features = OperationalPredictFeatureApplicator(
        # TODO use saved feature_names (should be saved while training the model)
        horizons=[pj["resolution_minutes"] / 60.0],
        feature_names=model.feature_names,
        feature_modules=model_specs.feature_modules,
    ).add_features(validated_data)

    # Prep forecast input by selecting only the forecast datetime interval (this is much smaller than the input range)
    # Also drop the load column
    forecast_start, forecast_end = generate_forecast_datetime_range(data_with_features)
    forecast_input_data = data_with_features[forecast_start:forecast_end].drop(
        columns="load"
    )

    # Check if sufficient data is left after cleaning
    if not validation.is_data_sufficient(
        data_with_features, pj["completeness_treshold"], pj["minimal_table_length"]
    ):
        logger.warning(
            "Using fallback forecast",
            forecast_type="fallback",
            pid=pj["id"],
            fallback_strategy=fallback_strategy,
        )
        forecast = generate_fallback(data_with_features, input_data[["load"]])

    else:
        # Predict
        model_forecast = model.predict(forecast_input_data)
        forecast = pd.DataFrame(
            index=forecast_input_data.index, data={"forecast": model_forecast}
        )

    # Add confidence
    forecast = ConfidenceIntervalApplicator(
        model, forecast_input_data
    ).add_confidence_interval(forecast, pj)

    # Prepare for output
    forecast = add_prediction_job_properties_to_forecast(
        pj,
        forecast,
        algorithm_type=str(model.path),
    )

    return forecast
