# gnns_for_gdp
Predicting GDP using graph neural networks on trade data

## Instructions

### Setup
1. Clone this repository with
  - ```git clone https://github.com/pboennig/gnns_for_gdp.git```
2. Install [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if preferred
  - Create the environment ```gdp-env``` with ```conda create --name gdp-env```
3. Install necessary dependencies:
  - ``` pip install -r requirements.txt ```

### Running Models
1. Navigate to your local copy of this repository (```cd gnns_for_gdp```)
2. Load in and pre-process the BACI and World Bank datasets
  - ```python create_year_file.py```
  - This step will take some time due to the sheer amount of data, but you'll be able to view the script's progress.
3. Create and run the two GATs
  - ```python run.py```
  - This step will also take some time. Neural network go brrrr...
4. Plot the loss and predictions of our two models
  - ```python plot.py```

## Structure of Repository
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
