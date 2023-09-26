# See README.md for docker-compose information
# Rebuild and run when code in /usaspending-api changes

FROM python:3.7.13

ARG CODE_HOME=/data-act/backend

WORKDIR $CODE_HOME

##### Install PostgreSQL client (psql)
RUN apt-get -y update && apt-get install -y postgresql-client

##### Install Required Python Packages
COPY requirements/ $CODE_HOME/requirements/
RUN python3 -m pip install .

##### Copy the rest of the project files into the container
COPY . $CODE_HOME

##### Ensure Python STDOUT gets sent to container logs
ENV PYTHONUNBUFFERED=1
