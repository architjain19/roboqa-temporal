# User Design Document

## Who are the users?
1. Engineering Interns
2. Working robotics engineers
3. Students new to robotics or data collection
4. Researchers and expert ML/Robotics users
5. Students working on Autonomous vehicle RL projects
6. Software managers and QA testers

## What do users need from the system?
1. **Reduce Manual Work**  
   Users collect hours of multi-sensor data (LiDAR, camera, IMU).  
   They cannot manually inspect every point-cloud frame or timestamp.  
   They need an automated quality analysis tool.

2. **Provide Clear, Actionable Quality Reports**  
   All users — especially beginners — need:  
   - simple visualizations  
   - clear explanations  
   - “good vs. bad” flags  
   - concrete suggestions (e.g., calibration issues, sync drift, corrupted frames)  
   Beginners need simple descriptions.  
   Experts need technical depth.

3. **Work With Multi-Sensor Data Easily**  
   Users need:  
   - drag-and-drop ROS bag processing  
   - automated extraction of camera, LiDAR, IMU topics  
   - no complex commands or setup

4. **Explain Data Quality in an Understandable Way**  
   Beginners need:  
   - explanations of what “quality”, “noise”, “drift”, “sync” means  
   - simple summaries and visual guides  
   Experts need:  
   - detailed metrics  
   - reproducible logs  
   - exportable evidence for papers or QA reports

5. **Support High Performance and Scalability**  
   Large robotics datasets (10–500 GB) must run efficiently:  
   - batch processing  
   - multi-threading  
   - GPU acceleration  
   - ability to run overnight without crashing

## What frustrations or pain points do users currently have?
1. **Manual inspection is too slow and unreliable**  
   Users cannot:  
   - manually inspect thousands of LiDAR frames  
   - spot timestamp drift  
   - detect calibration issues  
   - find corrupted segments

2. **Domain Expertise Required**  
   Beginners don’t know:  
   - what good data looks like  
   - how to interpret point-cloud noise  
   - how to read LiDAR intensity patterns  
   - how to judge drift or sensor dropouts

3. **No existing tool provides full multi-sensor QA**  
   Users use:  
   - ROS bag tools  
   - Open3D viewers  
   - Jupyter scripts  
   But none give all quality metrics in one place.

4. **Datasets are huge, repetitive, and messy**  
   Engineers collect:  
   - hours of driving  
   - multiple sensors  
   - repeated patterns  
   They need summaries.

## What behaviors and workflows does the system support?
1. **Beginner workflow**  
   - Upload bag file  
   - Click “Analyze”  
   - Receive:  
     - simple visual plots  
     - warnings  
     - quality score  
     - “corrupted segments” list  
   - No deep understanding required.

2. **Engineer workflow**  
   - Select specific sensors to analyze  
   - Inspect:  
     - timestamp drift  
     - calibration errors  
     - corrupted point-clouds  
     - frame drops  
   - Export a detailed report for team documentation.

3. **Researcher workflow**  
   - Run large-scale analysis  
   - Get reproducible metrics  
   - Export results (JSON/CSV/PDF)  
   - Include quality metrics in research papers.

4. **RL student workflow**  
   - Analyze dataset distribution:  
     - “good events”  
     - “bad events”  
   - Identify:  
     - failure cases  
     - rare events  
     - imbalance in data  
   - Decide training strategy based on quality.

5. **QA manager workflow**  
   - Stress test tool with large datasets  
   - Check accuracy and performance  
   - Report bugs and bottlenecks  
   - Ensure scalability and reliability.