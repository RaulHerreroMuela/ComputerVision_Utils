from monicas.models.autoencoder_to_regression_tflite import AutoencoderToRegressionTFlite
from monicas.models.foam_percentage_segmentation_tflite import FoamSegmentationTFLite
from monicas.models.EfficieNet_reg_and_classif_tflite import EfficientnetModelTFlite
from monicas.models.autoencoder_anomaly_tflite import AutoencoderAnomalyTFlite


class ModelLoader:
    def load_model(self, config):
        """Carga el modelo basado en el tipo especificado en el YAML."""
        model_type = config['model_type']

        if model_type == 'AutoencoderAnomaly':
            return AutoencoderAnomalyTFlite(id=config['id'], model_path=config['autoencoderAnomaly_path'])
        elif model_type == 'Classifier':
            return EfficientnetModelTFlite(
                id=config['id'],
                model_path=config['classifier_path'],
                activation_last_layer=config['activation_last_layer'],
                num_classes=config['num_classes']
            )
        elif model_type == 'Regressor':
            return EfficientnetModelTFlite(
                id=config['id'],
                model_path=config['pure_regressor_path'],
                activation_last_layer=config['activation_last_layer']
            )
        elif model_type == 'FoamSegmentation':
            return FoamSegmentationTFLite(
                path_image=config['path_image'],
                input_size=config['input_shape'],
                model_path=config['seg_model_path'],
                roi_mask_path=config['mask_path'],
                normalize=config['normalize'],
                mask_th=config['mask_threshold'],
                percent_jump=config['percent_jump']
            )
        elif model_type == 'AutoencoderRegressor':
            return AutoencoderToRegressionTFlite(
                id=config['id'],
                autoencoder_path=config['autoencoder_path'],
                regressor_path=config['regressor_path'],
                parameters=config['selected_features']
            )
        else:
            raise ValueError(f"Tipo de modelo desconocido: {model_type}")