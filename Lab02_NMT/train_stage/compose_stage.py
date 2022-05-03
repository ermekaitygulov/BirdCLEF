class ComposeStage:
    def __init__(self, stage_list):
        self.stage_list = stage_list

    def train(self, train_iterator, val_iterator):
        for stage in self.stage_list:
            stage.train(train_iterator, val_iterator)