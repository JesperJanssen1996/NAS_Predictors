class BasePredictor:
    def __init__(self,model_name):
        self.model_name = model_name
        self.model = None
        
    def pretrain(self,x_pretrain,y_pretrain):
        raise(NotImplementedError("Pretraining must be implemented in child class"))
        
    def train(self,x_train,y_train):
        raise(NotImplementedError("training must be implemented in child class"))    
        
    def predict(self,x_test):
        raise(NotImplementedError("predict must be implemented in child class"))    
        