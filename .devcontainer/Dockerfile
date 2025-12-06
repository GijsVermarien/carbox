FROM python:3.13-slim

# Install git, curl, sudo, build-essential, and other essential tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    sudo \
    build-essential \
    locales \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*


# Install Node for Gemini CLI.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @google/gemini-cli

# Set locale
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Create a non-root user
RUN useradd -m -s /bin/bash carbox && \
    echo 'carbox ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Set working directory
WORKDIR /workspace/carbox

COPY requirements.txt /tmp/pip-tmp/
RUN pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

# Switch to non-root user
USER carbox

# Add tools directory to PATH for python user
ENV PATH="/home/carbox/.local/bin:/workspace/carbox/.venv/bin:${PATH}"

# Keep container running
CMD ["sleep", "infinity"]
