:: `execute_python.bat` contains code that allows the user to run `unet.py` on the cluster.

:START

::Switch to Anaconda Shell and have direct excess 
call C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3

::Activate your Anaconda environment to install packages
call conda activate %UserProfile%\.conda\envs\pytorch_cpu

::Execute your python file in a specific path
python unet.py > output_unet.txt

:EOF