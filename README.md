# Mamta-Sharma_Girl-Hackathon_Silicon_2025

        GH-2025-Silicon-Engineering : DeepLogicAI: AI-Based Combinational Logic Depth Prediction
        
DeepLogicAI is an AI-powered tool designed to predict the combinational logic depth of RTL signals before synthesis. It helps hardware designers and verification engineers quickly identify timing violations without requiring full synthesis.

     Repository Structure
        📂 DeepLogicAI/
             ├── extract_features.py  # RTL Parsing & Feature Extraction using Yosys
             ├── train_model.py       # ML Model Training (Random Forest/XGBoost)
             ├── predict_depth.py     # Prediction on New RTL Modules
             ├── training_dataset.csv # Dataset for ML training
             ├── trained_model.pkl    # Pre-trained ML model
             │── README.md                 # Instructions on running the code
             │── requirements.txt          # List of dependencies
    1. Environment Setup
    
            Prerequisites
            Before running the code, install the following:

                    ---Python 3.8+
                    ---Yosys (for RTL parsing)
                    ---Required Python libraries


                git clone https://github.com/yourusername/DeepLogicAI.git                   #Clone the Repository
                cd DeepLogicAI
               python3 -m venv venv                                                         #Install Dependencies
               source venv/bin/activate  # On Windows: venv\Scripts\activate 
               pip install -r requirements.txt
               sudo apt install yosys  # Ubuntu                                             #Install Yosys
               brew install yosys  # macOS




               2. How to Run the Code
               
               ----- Extract Features from RTL Code (Provide an RTL file (module.v) and extract circuit features using Yosys.)
                             python extract_features.py --rtl data/module.v --output data/features.csv


                -----Train the Machine Learning Model (Use historical datasets to train the model.)
                           python train_model.py --data data/training_dataset.csv --output models/trained_model.pkl
    

                -------Predict Combinational Depth for New RTL (Predict logic depth for a new RTL design.)
                
                        python predict_depth.py --rtl data/module.v --model models/trained_model.pkl


3. Additional Information

Dataset Used: Benchmark datasets like ISCAS, OpenCores, MCNC.

ML Model: Random Forest/XGBoost Regression






The approach used to generate the algorithm


            Our approach consists of three main steps: Feature Extraction, Machine Learning Model Training, and Prediction for New RTL Designs. Each step is carefully 
             designed to ensure accuracy and efficiency.


             
            1. Feature Extraction from RTL
            Parsing the RTL Code: We use Yosys, an open-source synthesis tool, to convert the Verilog/VHDL RTL code into a gate-level netlist. This provides a structural 
            representation of the circuit.
            
            Building a Directed Acyclic Graph (DAG): The gate-level netlist is represented as a DAG, where:

            
            Nodes represent logic gates (AND, OR, NOT, etc.).
            
            Edges represent connections between gates.
            
            Extracting Key Features: Using graph traversal techniques, we extract relevant features such as:
            
            Fan-in/Fan-out: Number of input/output connections for each gate.
            
            Gate Count: Total number of logic elements in the design.
            
            Longest Combinational Path: Determined using topological sorting and dynamic programming to compute the longest path in the DAG.
            
            
            2. Machine Learning Model Training
            Dataset Preparation: We use benchmark datasets (e.g., ISCAS, OpenCores, MCNC) consisting of RTL designs with known logic depths from synthesis reports.
            
            Feature Engineering: Normalize extracted features and encode categorical variables where necessary.
            
            Model Selection: We train a Random Forest/XGBoost regression model, which learns the mapping between extracted features and combinational depth.
            
            Training Process:
            
            Perform hyperparameter tuning (e.g., number of trees, max depth) to optimize accuracy.
            
            Use cross-validation to prevent overfitting.
            
            Evaluate the model using Mean Squared Error (MSE) and R² score.
            
            
            3. Prediction for New RTL Designs
            
            Given a new RTL module, we repeat Step 1 (Feature Extraction) to obtain its circuit characteristics.
            
            The trained ML model then predicts the combinational logic depth in milliseconds, significantly reducing the time required compared to full synthesis.
            
            The result helps designers quickly assess timing risks and refine RTL architectures before detailed implementation.
            
            This AI-driven approach ensures fast and accurate depth estimation, improving timing closure efficiency in digital design workflows.
            

            
Proof of Correctness




                    To verify the correctness of our AI-based approach, we use the following validation methods:
                    
                    1. Ground Truth Comparison
                    
                    We run traditional synthesis tools (Yosys, Synopsys Design Compiler, or Cadence Genus) to extract the actual combinational logic depth of signals.
                    
                    Our predicted depth is compared with the true depth obtained from synthesis reports.
                    
                    2. Model Evaluation Metrics
                    
                    We use standard regression metrics to measure the accuracy of our predictions:
                    
                    Mean Absolute Error (MAE): Measures average prediction error.
                    
                    Root Mean Squared Error (RMSE): Penalizes larger errors more heavily.
                    
                    R² Score: Measures how well our model explains variance in logic depth.
                    
                    3. Cross-Validation
                    
                    We perform k-fold cross-validation on training data to avoid overfitting and ensure generalizability.
                    
                    4. Functional Testing on New RTL Modules
                    
                    The model is tested on unseen RTL designs with varying complexities to check if the predicted depth is consistent and accurate.
                    
                    5. Sensitivity Analysis
                    
                    We analyze how changes in fan-in, fan-out, and gate count affect predictions to ensure that the model correctly interprets logic dependencies.
                    
                    By systematically comparing AI-predicted results with actual synthesis reports and using robust evaluation metrics, we establish the correctness and 
                    reliability of our approach.


                    
Complexity Analysis


                    The overall complexity of our AI-based combinational depth prediction approach can be broken down into three main components
                    
                    1. Feature Extraction Complexity
                    
                    Parsing RTL and Generating Netlist:
                    
                    Using Yosys to extract gate-level netlists takes O(n log n), where n is the number of logic gates.
                    
                    Building the Directed Acyclic Graph (DAG):
                    
                    Representing the netlist as a graph has a complexity of O(V + E), where V is the number of logic elements and E is the number of connections.
                    
                    Finding Longest Path (Combinational Depth):
                    
                    Using topological sorting and dynamic programming, the longest path can be computed in O(V + E).
                    
                    Overall Feature Extraction Complexity: O(n log n) + O(V + E)
                    
                    2. Machine Learning Model Complexity
                    
                    Training Complexity:
                    
                    Random Forest: O(d × n log n), where d is the number of trees and n is the training data size.
                    
                    XGBoost: O(d × n) due to gradient boosting optimizations.
                    
                    Inference Complexity:
                    
                    Random Forest/XGBoost: O(1) for small models, O(d log n) for deeper models.
                    
                    Overall Training Complexity: O(d × n log n)
                    
                    Overall Inference Complexity: O(1) to O(d log n)
                    
                    3. Comparison with Traditional Synthesis
                    
                    Full Synthesis (Baseline Comparison):
                    
                    Traditional synthesis tools like Synopsys Design Compiler or Cadence Genus take minutes to hours (exponential in worst cases).
                    
                    AI-Based Prediction:
                    
                    Our approach reduces this to milliseconds, offering significant speed-up.
                    
                    Traditional Synthesis Complexity: O(2ⁿ) (Exponential in worst case)
                    
                     AI-Based Prediction Complexity: O(n log n) + O(1) (Much faster)
                     
                    Key Takeaways
                    
                     Feature extraction scales efficiently with circuit size (O(n log n)).
                     
                    Machine learning inference is nearly constant-time (O(1) or O(d log n)).
                    
                     Dramatic speed-up over traditional synthesis, making it practical for large-scale SoC/IP designs.
                     
                    This optimized complexity ensures that our approach is both fast and scalable for modern VLSI design workflows.
                    




                    
