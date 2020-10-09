from tensorboardX import SummaryWriter
import numpy as np
import csv
import sys
import os


class Visualise:
    class IterationStorage:
        class Train:
            def __init__(self):
                self.policy_loss = None  # type: float
                self.wm_loss = None  # type: float
                self.encoder_loss = None  # type: float
                self.unique_states = None  # type: float
                self.int_rewards = None  # type: float
                self.ext_rewards = None  # type: float
                self.folder = "Training/"

        class Eval:
            def __init__(self):
                self.wm_loss = None  # type: float
                self.int_rewards = None  # type: float
                self.ext_rewards = None  # type: float
                self.density_map = None  # type: np.ndarray
                self.pe_map = None  # type: np.ndarray
                self.q_map = None  # type: np.ndarray
                self.walls_map = None  # type: np.ndarray
                self.folder = "Evaluation/"

        def __init__(self):
            self.train = self.Train()
            self.eval = self.Eval()

        def get_train_iteration_data(self) -> dict:
            items = {self.train.folder + "Policy loss": self.train.policy_loss,
                     self.train.folder + "WM loss": self.train.wm_loss,
                     self.train.folder + "Encoder loss": self.train.encoder_loss,
                     self.train.folder + "Unique states visited": self.train.unique_states,
                     self.train.folder + "Mean ep intrinsic rewards": self.train.int_rewards,
                     self.train.folder + "Mean ep extrinsic rewards": self.train.ext_rewards
                     }
            return {key: value for key, value in items.items() if value is not None}

        def get_eval_iteration_data(self) -> dict:
            items = {self.eval.folder + "WM loss": self.eval.wm_loss,
                     self.eval.folder + "Mean ep intrinsic rewards": self.eval.int_rewards,
                     self.eval.folder + "Mean ep extrinsic rewards": self.eval.ext_rewards,
                     self.eval.folder + "Visitation densities": self.eval.density_map,
                     self.eval.folder + "Prediction error map": self.eval.pe_map,
                     self.eval.folder + "Q-value map": self.eval.q_map,
                     }
            return {key: value for key, value in items.items() if value is not None}

    def __init__(self, run_name, **kwargs):
        super(Visualise, self).__init__()
        self.vis_args = kwargs
        self.folder_name = run_name
        os.makedirs(run_name, exist_ok=True)
        self.writer = SummaryWriter(self.folder_name)
        self.train_interval = kwargs['export_interval']
        self.eval_interval = kwargs['eval_interval']
        self.train_id = 0  # train count for visualisation
        self.eval_id = 0  # valid count for visualisation
        with open(f'{self.folder_name}/params{kwargs["time_stamp"]}.csv', mode='w') as f:
            csv_writer = csv.writer(f)
            for key, value in kwargs.items():
                csv_writer.writerow([key, value])
        self.storage = self.IterationStorage()

    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        assert len(img.shape) == 1 or len(img.shape) == 3, 'Image must have channel dimension'
        if img.shape[0] == 1:
            min = np.amin(img, axis=(1, 2))
            max = np.amax(img, axis=(1, 2))
        elif img.shape[0] == 3:
            min = np.amin(img, axis=(1, 2))[:, np.newaxis, np.newaxis]
            max = np.amax(img, axis=(1, 2))[:, np.newaxis, np.newaxis]
        else:
            raise AssertionError(f'Image must be Grayscale or RGB, but found channel dim was of size {img.shape[0]}')
        # return np.sqrt(img / (max + 1e-8))
        new_img = img / max if max != 0.0 else img
        return new_img

    def train_iteration_update(self, t: int = None):
        if t is None:
            self.train_id += self.train_interval
            t = self.train_id
        for key, value in self.storage.get_train_iteration_data().items():
            self.writer.add_scalar(key, value, t)

    def eval_iteration_update(self, t: int = None):
        if t is None:
            self.eval_id += self.eval_interval
            t = self.eval_id
        for key, item in self.storage.get_eval_iteration_data().items():
            if isinstance(item, float):
                self.writer.add_scalar(key, item, t)
            elif isinstance(item, np.ndarray):
                if len(item.shape) == 2:
                    item = np.expand_dims(item, axis=0)
                item = self.normalize(item.clip(max=0.01))
                if self.storage.eval.walls_map is not None:
                    item = np.concatenate((item, item + self.storage.eval.walls_map, item), axis=0)
                self.writer.add_image(key, item, t)
            else:
                raise TypeError(f'Received item "{key}" is of wrong object type. Expected "float" or "np.ndarray", '
                                f'received "{type(item)}"')

    def close(self):
        self.writer.close()
