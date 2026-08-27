# start os-world server
sudo groupadd docker 
sudo gpasswd -a $USER docker
source activate tonggui
python desktop_env/docker_server/server.py 
