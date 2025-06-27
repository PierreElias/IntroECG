FROM python:3.8
WORKDIR /opt/cradlenet/

RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY . .

RUN poetry install --no-interaction --no-ansi

RUN pip install torch==2.0.1 torchvision==0.15.2

VOLUME ["/xml_input", "/processed_data", "/results"]

CMD ["/bin/sh", "runall.sh"]


