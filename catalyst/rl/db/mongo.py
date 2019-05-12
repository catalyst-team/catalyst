import datetime
import pymongo
from catalyst.utils.compression import pack, unpack
from .core import DBSpec


class MongoDB(DBSpec):
    def __init__(self, port=12000, prefix=None, sync_epoch=False):
        self._server = pymongo.MongoClient(host="127.0.0.1", port=port)
        self._prefix = "" if prefix is None else prefix

        self._shared_db = self._server["shared"]
        self._agent_db = self._server[f"agent_{self._prefix}"]

        self._trajectory_collection = self._shared_db["trajectories"]
        self._weights_collection = self._agent_db["weights"]
        self._flag_collection = self._agent_db["flag"]
        self._last_datetime = datetime.datetime.min

        self._epoch = 0
        self._sync_epoch = sync_epoch

    @property
    def num_trajectories(self) -> int:
        num_trajectories = self._trajectory_collection.count() - 1
        return num_trajectories

    def set_sample_flag(self, sample: bool):
        self._flag_collection.replace_one(
            {"prefix": "sample_flag"},
            {
                "sample_flag": sample,
                "prefix": "sample_flag"
            },
            upsert=True
        )

    def get_sample_flag(self) -> bool:
        flag_obj = self._flag_collection.find_one(
            {"prefix": {"$eq": "sample_flag"}}
        )
        flag = int(flag_obj.get("sample_flag") or -1) == int(1)
        return flag

    def push_trajectory(self, trajectory):
        trajectory = pack(trajectory)
        self._trajectory_collection.insert_one({
            "trajectory": trajectory,
            "date": datetime.datetime.utcnow(),
            "epoch": self._epoch
        })

    def get_trajectory(self, index=None):
        assert index is None

        trajectory_obj = self._trajectory_collection.find_one(
            {"date": {"$gt": self._last_datetime}}
        )
        if trajectory_obj is not None:
            self._last_datetime = trajectory_obj["date"]

            trajectory, trajectory_epoch = \
                unpack(trajectory_obj["trajectory"]), trajectory_obj["epoch"]
            if self._sync_epoch and self._epoch != trajectory_epoch:
                trajectory = None
        else:
            trajectory = None

        return trajectory

    def dump_weights(self, weights, prefix, epoch):
        self._epoch = epoch

        weights = pack(weights)
        self._weights_collection.replace_one(
            {"prefix": prefix},
            {
                "weights": weights,
                "prefix": prefix,
                "epoch": self._epoch
            },
            upsert=True
        )

    def load_weights(self, prefix):
        weights_obj = self._weights_collection.find_one({"prefix": prefix})
        weights = weights_obj.get("weights")
        if weights is None:
            return None
        self._epoch = weights_obj["epoch"]
        weights = unpack(weights)
        return weights
