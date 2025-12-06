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


# Install Node.js 22.x LTS (includes npm)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install @anthropic-ai/claude-code globally
RUN npm install -g @google/gemini-cli

# Set locale
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Create a non-root user
RUN useradd -m -s /bin/bash researcher && \
    echo 'researcher ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER researcher

# Add tools directory to PATH for python user
ENV PATH="/home/python/.local/bin:/workspace/.venv/bin:${PATH}"

# Keep container running
CMD ["sleep", "infinity"]
