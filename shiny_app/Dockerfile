FROM rocker/shiny-verse:4.4.1

# System dependencies for R packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages
RUN install2.r --error --skipinstalled \
    shiny \
    bslib \
    DT \
    plotly \
    visNetwork \
    dplyr \
    readr \
    jsonlite \
    arrow \
    tidyr \
    stringr

# Copy app
COPY . /srv/shiny-server/glyco-kg/

WORKDIR /srv/shiny-server/glyco-kg

EXPOSE 3838

CMD ["R", "-e", "shiny::runApp('.', host='0.0.0.0', port=3838)"]
