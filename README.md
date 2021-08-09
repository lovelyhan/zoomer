# Zoomer

Implementation of the method proposed in the paper:

*Zoomer: Improving and Accelerating Recommendation on Web-Scale Graphs via Regions of Interests.*

## Enviroment Requirement

The code has been tested running under Python 3.6.5. The requireed packages are as follows:

- ```XDL == 2.1```
- ```Euler == 2.0```

## Configuration

To apply Zoomer to your own data, please:

- Configure ```model/xdl2_runner_zoomer_base_train.json```.
- Set the hyperparameters of the model in ```model/zoomer_model_config.py```.
- ROI-based Multi-Level Attention in this paper contains three major attention mechanisms: a flexible node-level feature embedding projection, an edge-level neighbor reweighing, and a semantic-level relation combination. you can enable or disable these attentions in ```model_option```.

## Data

- We used Alibaba's internal data and built our own industrial dataset in the experiment. Due to data security reasons, we can not disclose the data we use.
- You can apply Zoomer to your own datasets by following our input dat's format. 
- We conduct our experiment on three heterogeneous graph of different scales:
  - million-scale-graph: contains 2 million nodes, 400 million edges;
  - hundred-billion-scale-graph: contains 120 million, 30 billion edges;
  - million-scale-graph: contains 1.2 billion nodes, 260 billion edges.

### Input Data Format

- User log data input

| Field    | Type    | Comment                                                      |
| -------- | ------- | ------------------------------------------------------------ |
| query_id | int64   | ID of the query                                              |
| user_id  | int64   | ID of the user                                               |
| item_id  | int64   | ID of the item                                               |
| labels   | float32 | value equals to 0 or 1, indicating whether user clicks item under the given query. |

- Graph data

| Field        | Type   | Comment                                                      |
| ------------ | ------ | ------------------------------------------------------------ |
| node_id      | int64  | node ID in the graph                                         |
| node_type    | string | type of the node in the graph, we use q(query), u(user) and i(item) |
| edge_type    | string | type of the edge in the graph                                |
| feature_name | string | name for the feature of nodes, we use different features for different type of nodes |

## Run the Code

To try our code,  you can input your own data,  and directly run the following command:

```
python xdl2_runner_zoomer_base_train.py
```

## References

- [XDL](https://github.com/alibaba/x-deeplearning)
- [Euler](
