# VisualLoc

VisualLoc is a library built to unify the research in the field of visual place recognition.
It is the case that many state of the art algorithms reported in the literature are deployed
on slightly different configured datasets, making it difficult to determine state of the art. 
This repositoy implements over 10 of the best Visual Place Techniques currently available along 
with 10 benchmarks so that their performances can be accurately compared. 

The repository also provides a framework to build, design, train, deploy and evaluate new 
visual place recognition methods, thereby serving the research communities in visual navigation 
and place recognition.



## Usage 
The two main concepts in this repository are "methods" and "datasets" each can be instantaited 
as objects and have a number of member functions useful for visual place recognition tasks.
A workflow for using a method for place recognition can be shown below. 


```python
# Import the method
from VisualLoc.Methods import SampleMethod

# Create an instance of the method
method = SampleMethod()

# Load dataset
dataset = 'path_to_dataset'

# Perform place recognition
results = method.recognize_place(dataset)

# Display results
print(results)
```