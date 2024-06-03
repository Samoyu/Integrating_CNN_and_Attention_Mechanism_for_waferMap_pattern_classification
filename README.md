# Integrating CNN and Attention Mechanism for waferMap pattern classification
## Prerequisites

- Python 3.x
- Download waferMap data on this [site](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) and named the file **"dataset.pkl"** in your folder

## Installation
Follow these steps to set up and run the project:

1. **Clone the Repository**
   ```
   git clone https://github.com/Samoyu/DANet_opt_experiments.git
   ```

2. **Set up a Python Virtual Environment**
   ```
   python3 -m venv venv
   source venv/bin/activate  # Use `venv\Scripts\activate` on Windows
   ```

3. **Install Required Dependencies**
   ```
   pip install -r requirements.txt
   ```


## Usage

To run the script, use the following commands:

```
python3 main.py --size 224 \
                --batch_size 16 \
                --seed 60 \
                --optimizer adam \
                --epochs 10 \
                --learning_rate 1e-4 \
                --in_channels 512 \
                --out_channels 128 \
                --out_dim 8
```