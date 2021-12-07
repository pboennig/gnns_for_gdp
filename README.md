# gnns_for_gdp
Predicting GDP using graph neural networks on trade data

## Repo structure
- output/
  - Contains data for training, validating, and testing model
- results/
  - Model predictions and losses over the course of training, validation, and testing
- world_bank_gdp/
  - Datasets used for this project
  - emp/
    - National unemployment rates (1995 - 2019)
  - cpi/
    - National inflation rates (1995 - 2019)
  - gdp/
    - National gross domestic product (GDP) (1995 - 2019)
  - pop/
    - National population statistics (1995 - 2019)
- baseline.py
  - Two-layer graph neural network using Graph Attention Network (GAT) for its convolutional layers
  - Does not incorporate edge features
- model.py
  - Two-layer graph neural network using GAT for convolutional layers
  - Incorporates edge features
- create_year_file.py
  - Reads in a pre-processes datasets from world_bank_gdp/
- plot.py
  - Plots model loss and predictions
- run.py
  - Trains, validates, and tests model
- util.py
  - An assortment of useful functions for this project
