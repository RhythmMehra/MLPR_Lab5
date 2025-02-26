FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install nbconvert jupyter wandb numpy pandas scikit-learn opencv-python matplotlib scipy
CMD ["python", "RhythmLab5.py"]