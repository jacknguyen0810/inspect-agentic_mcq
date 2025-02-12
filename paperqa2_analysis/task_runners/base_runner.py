# Import libraries

class BaseRunner:

    def __init__(
        self,
        model: str,
        data: str,
        eval_metric,
        api_key = None 
    ):
        
        self.data = data
        self.model = model
        self.eval_metric = eval_metric
        self.api_key = api_key