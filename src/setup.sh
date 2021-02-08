#Check version
python3 --version
#set an alias of python3 for python
alias python='python3'
source ~/.bash_profile

#preset for pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
python3 -m pip install --upgrade pip

#Set dependencies
python3 -m pip install TensorFlow=='2.3.0'
python3 -m pip install pandas
python3 -m pip install numpy
python3 -m pip install scikit-learn=='0.23.2'
python3 -m pip install matplotlib
python3 -m pip install pickle5=='4.0'

echo "You are all set!"
