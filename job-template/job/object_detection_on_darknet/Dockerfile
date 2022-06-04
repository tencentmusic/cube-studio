FROM ubuntu:18.04
WORKDIR /app
RUN apt update && apt install -y gcc make git g++ wget zip
RUN git clone https://github.com/pjreddie/darknet.git
RUN cd darknet && sed -i 's@OPENMP=0@OPENMP=1@g' Makefile && make
COPY setup_args.py /app
COPY launcher.sh /app
RUN wget https://pengluan-76009.sz.gfp.tencent-cloud.com/github/yolov3.weights

RUN wget https://pengluan-76009.sz.gfp.tencent-cloud.com/github/coco_data_sample.zip
RUN unzip coco_data_sample.zip && cd coco_data_sample && bash reset_list.sh

RUN apt install -y python3.6-dev python3-pip libsasl2-dev libpq-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

RUN chmod 777 launcher.sh
ENTRYPOINT ["./launcher.sh"]
