FROM  pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
#FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

WORKDIR /workspace
RUN chmod -R a+w /workspace
ADD ./launcher.py /workspace/

ENTRYPOINT ["torchrun", "launcher.py"] 