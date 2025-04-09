# DWFS-Obfuscation
DWFS-Obfuscation: Dynamic Weighted Feature Selection for Robust Malware Classification under Obfuscation  Paper Source Code

## How to run this.

1. first conda envriment<br>
   cd DWFS-Obfuscation<br>
   pip install -r requirements.txt <br>
3. if (you want to train a new model）:<br>
       you need prepare a pair of unobfuscation malware and obfuscated malware.<br>
       then you get feature rank from DWFS folder script,<br>
       and get Pytorch Dataset from GNNModel.<br>
       use Trainer.py train your Model.<br>
   else:<br>
       cd ObfuscationAPKTest Folder<br>
       then run ObfuscationPerClassTest.py<br>
       DWFS-Obfuscation_Test_Data it include Dataset we open-source, only 50 per class and per obfuscation strategies.<br>
