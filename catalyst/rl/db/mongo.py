import datetime
import pymongo
from catalyst.utils.serialization import serialize, deserialize
from catalyst.utils.compression import compress, decompress, LZ4_ENABLED
from .core import DBSpec


class MongoDB(DBSpec):
    def __init__(self, port=12000, prefix=None, use_compression=True):
        self._server = pymongo.MongoClient(host="127.0.0.1", port=port)
        self._prefix = "" if prefix is None else prefix

        self._shared_db = self._server["shared"]
        self._agent_db = self._server[f"agent_{self._prefix}"]

        self._trajectory_collection = self._shared_db["trajectories"]
        self._weights_collection = self._agent_db["weights"]
        self._last_datetime = datetime.datetime.min

        self._use_compression = use_compression and LZ4_ENABLED
        if self._use_compression:
            self._pack = compress
            self._unpack = decompress
        else:
            self._pack = serialize
            self._unpack = deserialize

    @property
    def num_trajectories(self) -> int:
        num_trajectories = self._trajectory_collection.count() - 1
        return num_trajectories

    def push_trajectory(self, trajectory):
        trajectory = self._pack(trajectory)
        self._trajectory_collection.insert_one({
            "trajectory": trajectory,
            "date": datetime.datetime.utcnow()
        })

    def get_trajectory(self, index=None):
        assert index is None

        trajectory_obj = self._trajectory_collection.find_one(
            {"date": {"$gt": self._last_datetime}}
        )
        if trajectory_obj is not None:
            trajectory = trajectory_obj["trajectory"]
            trajectory = self._unpack(trajectory)
            self._last_datetime = trajectory_obj["date"]
        else:
            trajectory = None
        return trajectory

    def dump_weights(self, weights, prefix):
        weights = self._pack(weights)

        self._weights_collection.replace_one(
            {"prefix": prefix},
            {
                "weights": weights,
                "prefix": prefix
            },
            upsert=True
        )

    def load_weights(self, prefix):
        weights_obj = self._weights_collection.find_one({"prefix": prefix})
        weights = weights_obj.get("weights")
        if weights is None:
            return None
        weights = self._unpack(weights)
        return weights
