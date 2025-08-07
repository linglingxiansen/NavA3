# NavA3
Repository for NavA3: Understanding Any Instruction, Navigating Anywhere, Finding Anything

**Hardware:** RealMan and Realsense D435i Camera


## Installation


```
pip install openai sounddevice scipy opencv-python numpy 
```

## Api-key

Put your openai api-key in the gpt_english.py
```
key='your openai api-key'
```

## Run

1. Use 3d scanner app to construct your 3d scene and top-down map
   
2. Change the x_min, y_min, x_max, y_max for different rooms in the navia3_english.py
  
3. Connect your realman robot and run navia3_english.py
