

cp entrypoint.sh mindformers/
cp entrypoint.sh mindformers/research/qwen1_5/
cp predict_qwen1_5_7b_parallel.yaml mindformers/research/qwen1_5/
echo 'while pgrep -f python > /dev/null; do echo "process is running, watch log from outputpath/msrun_log/"; sleep 5; done; echo "No process found. Exiting."' >> mindformers/scripts/msrun_launcher.sh