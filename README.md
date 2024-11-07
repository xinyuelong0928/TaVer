# Lingual-Fusion Adapter-Based Transfer Learning for Low-Resource Code Vulnerability Detection

## Code repository for the study
This replication package contains the dataset and code for our paper `Lingual-Fusion Adapter-Based Transfer Learning for Low-Resource Code Vulnerability Detection`.The work proposes TaVer, an innovative framework that utilizes lingual-fusion adapters and temporal modeling to enhance transfer learning in low-resource code vulnerability detection. Designed to improve accuracy and efficiency in detecting vulnerabilities in languages with limited annotated data, TaVer leverages temporal modeling to capture potential risks as code evolves. This framework equips developers with powerful tools to identify and address security vulnerabilities in their codebases effectively.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/xinyuelong0928/TaVer.git
   cd TaVer

2. **Environment**:  
    Make sure you have Python 3.8.10 installed, then install the dependencies:
    
    ```bash
    pip install -r requirements.txt

## Usage
### Data Preparation
- **Datasets**: The dataset for this project is located in the `datasets/` directory and includes samples from multiple programming languages, such as C, Java, Python, and PHP. If using a new dataset, please ensure it aligns with the required format to facilitate effective vulnerability detection.

- **Data Processing**: 
  ```bash
  bash data.sh

### Training and Evaluation
     bash train.sh

### Testing
     bash test.sh

## File Structure
- `checkpoints/`: saved models
- `datasets/`: `.jsonl` format datasets
- `models/`: CodeBERT pre-trained model
- `results/`: code vulnerability detection results
- `utils/`: utility scripts and tools
   - `parserTool/`: code parsing tools
   - `pkl_folder/`: `.pkl` files generated from data preprocessing



