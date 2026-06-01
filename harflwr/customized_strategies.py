# Created with help of Claude for extending existing FedAvg to chunk serving logic
import json

from flwr.serverapp.strategy import FedAvg


class SequentialChunkFedAvg(FedAvg):

    def __init__(self, selection_counts, **kwargs):
        super().__init__(**kwargs)
        self.selection_counts = selection_counts

    def configure_train(self, server_round, arrays, config, grid):
        config["client-selection-counts"] = json.dumps(self.selection_counts)
        config["server-round"] = server_round
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        active_replies = []
        for msg in replies:
            if msg.has_content():
                if "metrics" in msg.content:
                    if msg.content["metrics"].get("num-examples", 0) > 0:
                        active_replies.append(msg)

        if len(active_replies) == 0:
            print("Round " + str(server_round) + ": All clients exhausted, skipping aggregation")
            return None, None

        return super().aggregate_train(server_round, active_replies)