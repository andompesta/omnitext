FROM andompesta1/pytorch-dev-nlp:1.5
MAINTAINER andompesta

ADD ./requirements.txt /workspace/requirements.txt

RUN pip install -r /workspace/requirements.txt

ENV ENV_FOR_DYNACONF=dev
