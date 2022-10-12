
apt install -y openssh-server openssh-client && echo root:cube-studio | chpasswd
echo "Port ${SSH_PORT}" /etc/ssh/sshd_config
sed -i "s/#PermitEmptyPasswords no/PermitEmptyPasswords yes/g" /etc/ssh/sshd_config
sed -i "s/#PermitRootLogin yes/PermitRootLogin yes/g" /etc/ssh/sshd_config
echo root:cube-studio | chpasswd
service ssh restart
