
pip3 install paddlepaddle paddleseg
conda install -y -c conda-forge dlib

python /app/download_hubman_seg_inference_models.py
python /app/download_animegan2_inference_models.py
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2 && rm shape_predictor_68_face_landmarks.dat.bz2

