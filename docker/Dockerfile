#Ubuntu with a desktop and VNC server.  Will allow us to use gui monitor tools that these libaries have such as tensorboard
FROM consol/ubuntu-xfce-vnc

#switches default user to root
USER 0

#use bash for the shell
SHELL [ "/bin/bash","-c" ]

#sets the environment so that we can use conda after install
ENV PATH="~/anaconda3/condabin:${PATH}"

#Used for GPU setup
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

#Needed to build some packages
RUN apt update \
&& apt install gcc -y \
&& apt install build-essential -y \
&& apt install unzip -y \
&& apt install nomacs -y

#gets the anaconda package, installs it, and adds it to the path
#gets the anaconda package
RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh \
#sets the correct permissions to allow install
&& chmod 777 Anaconda3-2019.03-Linux-x86_64.sh \
#silently installs the package
&& ./Anaconda3-2019.03-Linux-x86_64.sh -b  \
#sets the path, may not be needed with the ENV PATH being used above
&& echo "export PATH=\"/headless/anaconda3/condabin:$PATH\"">>.bashrc \
&& source ~/.bashrc

RUN conda init \
&& conda create -n intelligent python=3.7 \
&& conda install -n intelligent matplotlib=3.0.3 \
&& conda install -n intelligent scikit-learn \
&& conda install -n intelligent pandas \
&& conda install -n intelligent opencv \
&& conda install -n intelligent seaborn \ 
&& conda install -n intelligent tqdm 
