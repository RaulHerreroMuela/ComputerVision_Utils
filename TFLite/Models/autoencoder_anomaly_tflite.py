from tensorflow import lite


class AutoencoderAnomalyTFlite:

    def __init__(self, id, model_path=None):

        self.model_path = model_path
        self.interpreter = lite.Interpreter(model_path=model_path)
        self._id = id

    def predict(self, data):
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        X = data

        self.interpreter.set_tensor(input_details[0]['index'], X)
        self.interpreter.invoke()

        result = self.interpreter.get_tensor(output_details[0]['index'])
        return result
