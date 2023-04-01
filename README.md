# two_WL_NTK

scp -r hahn@remote.cip.ifi.lmu.de:~/masterarbeit/two_WL_NTK/Data/Preprocessed/DataLoader DataLoader

scp -r PTC_MR.zip hahn@remote.cip.ifi.lmu.de:~/masterarbeit/two_WL_NTK/Data/TUData

scp -r hahn@remote.cip.ifi.lmu.de:~/masterarbeit/two_WL_NTK/Data/Preprocessed/DataLoader.zip Data/Preprocessed
source Thesis/bin/activate


scp -r hahn@remote.cip.ifi.lmu.de:~/masterarbeit/two_WL_NTK/Data/Preprocessed/DataLoader.zip Data/Preprocessed


tmux new -d 'python GC_train.py > output_5.log'