# DWFS-Obfuscation
DWFS-Obfuscation: Dynamic Weighted Feature Selection for Robust Malware Classification under Obfuscation  Paper Source Code

## How to run this.

1. first conda envriment<br>
   cd DWFS-Obfuscation<br>
   pip install -r requirements.txt
3. if (you want to train a new model）:
       you need prepare a pair of unobfuscation malware and obfuscated malware.
       then you get feature rank from DWFS folder script,
       and get Pytorch Dataset from GNNModel.
       use Trainer.py train your Model.
   else:
       cd ObfuscationAPKTest Folder
       then run ObfuscationPerClassTest.py
       it include Dataset we open-source.
