import pandas as pd
from tensorflow import lite

class AutoencoderToRegressionTFlite:

    def __init__(self, id, autoencoder_path=None, regressor_path=None,parameters=[]):
        self.id = id
        self.parameters = parameters
        self.interpreter_autoencoder = lite.Interpreter(model_path=autoencoder_path)
        self.interpreter_regressor = lite.Interpreter(model_path=regressor_path)


    def predict(self, data):

        self.interpreter_autoencoder.allocate_tensors()
        self.interpreter_regressor.allocate_tensors()

        input_details_autoencoder = self.interpreter_autoencoder.get_input_details()
        input_details_regressor = self.interpreter_regressor.get_input_details()

        output_details_autoencoder = self.interpreter_autoencoder.get_output_details()
        output_details_regressor = self.interpreter_regressor.get_output_details()

        self.interpreter_autoencoder.set_tensor(input_details_autoencoder[0]['index'], data)
        self.interpreter_autoencoder.invoke()

        latent_parameters = self.interpreter_autoencoder.get_tensor(output_details_autoencoder[0]['index'])
        latent_parameters = latent_parameters.reshape(latent_parameters.shape[0], -1)

        df_bruto = pd.DataFrame(data=latent_parameters)
        df = df_bruto.iloc[:, self.parameters]

        self.interpreter_regressor.set_tensor(input_details_regressor[0]['index'], df)
        self.interpreter_regressor.invoke()

        result = self.interpreter_regressor.get_tensor(output_details_regressor[0]['index'])

        return result