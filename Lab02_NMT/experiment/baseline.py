from experiment import EXPERIMENT_CATALOG
from experiment.base import Experiment
from train_stage import MainStage, ComposeStage
from utils import add_to_catalog


@add_to_catalog('baseline', EXPERIMENT_CATALOG)
class Baseline(Experiment):
    def init_trainer(self):
        main_stage = MainStage(self.model, 'main_stage', self.device,
                               self.config['train'], self.metrics)
        sgd_train_stage = MainStage(self.model, 'sgd_stage', self.device,
                                    self.config['sgd_train'], self.metrics)
        stage = ComposeStage([main_stage, sgd_train_stage])
        return stage
