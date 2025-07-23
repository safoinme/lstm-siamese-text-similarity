# Use Kubeflow Jupyter TensorFlow CUDA Full image as base
FROM kubeflownotebookswg/jupyter-tensorflow-cuda-full:v1.8.0

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NLTK_DATA=/home/jovyan/nltk_data
ENV TRANSFORMERS_CACHE=/home/jovyan/.cache/transformers

# Install system dependencies
USER root

# Update package lists and install essential tools
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create directories for jovyan user with proper ownership
RUN mkdir -p /home/jovyan/.local/bin /home/jovyan/.config/uv /home/jovyan/.cache/transformers && \
    chown -R jovyan:users /home/jovyan/.local /home/jovyan/.config /home/jovyan/.cache

# Switch back to jovyan user (standard for Kubeflow notebooks)
USER jovyan

# Add uv to jovyan user's PATH
RUN echo 'export PATH="/home/jovyan/.local/bin:$PATH"' >> /home/jovyan/.bashrc
ENV PATH="/home/jovyan/.local/bin:$PATH"

# Install uv for jovyan user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /home/jovyan

# Copy requirements file if it exists
COPY --chown=jovyan:users requirements.txt* ./

# Install TensorFlow GPU and common packages for LSTM Siamese text similarity
RUN /home/jovyan/.local/bin/uv pip install --system \
    tensorflow-gpu \
    keras \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    nltk \
    spacy \
    transformers \
    datasets \
    sentence-transformers \
    gensim \
    wordcloud \
    plotly \
    tqdm \
    jupyter \
    jupyterlab \
    ipywidgets \
    tensorboard

# Install additional requirements if file exists
RUN if [ -f requirements.txt ]; then /home/jovyan/.local/bin/uv pip install --system -r requirements.txt; fi

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the local directory contents into the container
COPY --chown=jovyan:users . .

# Verify installations
RUN python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}'); print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"