# Zoomer

## Requirements

We conduct our experiment using the following two open source systems.

- ```XDL 2.1```
- ```Euler 2.0```

## Running the training

We run our training on X-DeepLearning (XDL). XDL is a framework for deep learning tasks of high-dimensional sparse data scenarios (such as advertising/recommendation/search, etc.)

```
python xdl2_runner_zoomer_base_train.py
```

## Configuration

- Set up your configuration in ```model/xdl2_runner_zoomer_base_train.json```.

- Set the hyperparameters of the model in ```model/zoomer_model_config.py```.

- In the paper, we use 3-level attention, namely feature-level, edge-level and semantic level, you can enable or disable these attentions in ```model_option```.

## Data

- We used Alibaba's internal data and built our own industrial dataset in the experiment. Due to data security reasons, we can not disclose the data we use.
- We conduct our experiment on three heterogeneous graph of different scales:
  - million-scale-graph: contains 2 million nodes, 400 million edges;
  - hundred-billion-scale-graph: contains 120 million, 30 billion edges;
  - million-scale-graph: contains 1.2 billion nodes, 260 billion edges.

### Type of data

- Input data

| Field          | Type    | Comment                                                      |
| -------------- | ------- | ------------------------------------------------------------ |
| query_id       | int64   | ID of the query                                              |
| user_id        | int64   | ID of the user                                               |
| item_id        | int64   | ID of the item                                               |
| labels         | float32 | value equals to 0 or 1, indicating whether user clicks item under the given query. |
| node_embedding | float32 | embedding for the node in embedding table                    |

- Graph data

| Field        | Type   | Comment                                                      |
| ------------ | ------ | ------------------------------------------------------------ |
| node_id      | int64  | node ID in the graph                                         |
| node_type    | string | type of the node in the graph, we use q(query), u(user) and i(item) |
| edge_type    | string | type of the edge in the graph                                |
| feature_name | string | name for the feature of nodes, we use different features for different type of nodes |

## 

## References

- [XDL](https://github.com/alibaba/x-deeplearning)
- [Euler](https://github.com/alibaba/euler)

