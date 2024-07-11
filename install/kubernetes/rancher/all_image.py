images = open("rancher-images-mini.txt").readlines()

images = list(set([x.strip() for x in images if x.strip()]))
# 通过私有仓库，将公有镜像下发到内网每台机器上，例如内网ccr.ccs.tencentyun.com的仓库，共约26G
harbor_repo = 'xx.xx.xx.xx:xx/xx/'
pull_file = open('pull_rancher_images.sh',mode='w')
push_harbor_file = open('push_rancher_harbor.sh',mode='w')
pull_harbor_file = open('pull_rancher_harbor.sh', mode='w')

pull_save_file = open('rancher_image_save.sh',mode='w')
load_image_file = open('rancher_image_load.sh',mode='w')

push_harbor_file.write('docker login '+harbor_repo[:harbor_repo.index('/')]+"\n")
pull_harbor_file.write('docker login '+harbor_repo[:harbor_repo.index('/')]+"\n")

for image in images:
    # print(image)
    # print(image)
    image = image.replace('<none>', '')
    new_image = harbor_repo + image.replace('rancher/', '').replace('/', '-')

    # 可联网机器上拉取公有镜像并推送到私有仓库
    # print('docker pull %s && docker tag %s %s && docker push %s &' % (image,image,image_name,image_name))
    push_harbor_file.write('docker pull %s && docker tag %s %s && docker push %s &\n' % (image,image,new_image,new_image))
    pull_save_file.write('docker pull %s && docker save %s | gzip > %s.tar.gz &\n' % (image, image, image.replace('/','-').replace(':','-')))

    # # # 内网机器上拉取私有仓库镜像
    # print("docker pull %s && docker tag %s %s &" % (image_name,image_name,image))
    pull_harbor_file.write("docker pull %s && docker tag %s %s &\n" % (new_image,new_image,image))
    load_image_file.write('gunzip -c %s.tar.gz | docker load &\n' % (image.replace('/','-').replace(':','-')))

    # # 拉取公有镜像
    # print("docker pull %s && docker tag %s %s &" % (image_name,image_name,image))
    print("docker pull %s &" % (image,))
    pull_file.write("docker pull %s &\n" % (image,))

pull_file.write('\nwait\n')
pull_save_file.write('\nwait\n')
load_image_file.write('\nwait\n')
pull_harbor_file.write('\nwait\n')
push_harbor_file.write('\nwait\n')

print('若服务器可以链网，直接执行sh pull_rancher_images.sh')
print('若服务器无法联网，替换本代码中的内网harbor仓库名，先在可联网机器上执行push_harbor.sh，再在内网机器上执行pull_harbor.sh')







