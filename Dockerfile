FROM docker.yfish.x/yfish-pytorch-lightweight
RUN pip3 install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install molsets -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install torchsnooper -i https://pypi.tuna.tsinghua.edu.cn/simple
ADD src /opt/fergie/src
ENV PATH="/opt/fergie/src:$PATH"
