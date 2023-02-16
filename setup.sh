#! /bin/bash

pip3 install --upgrade pip setuptools wheel
/SequenceR/src/setup_env.sh
data_path=/SequenceR/data
OpenNMT_py=/SequenceR/src/lib/OpenNMT-py
apt-get install -y libcam-pdf-perl
PERL_MM_USE_DEFAULT 1
git clone https://github.com/rjust/defects4j /SequenceR/src/lib/defects4j
cpan App::cpanminus
cpanm --installdeps /SequenceR/src/lib/defects4j/
/SequenceR/src/lib/defects4j/init.sh
PATH="${PATH}:/SequenceR/src/lib/defects4j/framework/bin"
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html