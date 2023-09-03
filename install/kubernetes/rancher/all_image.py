images = open("rancher-images.txt").readlines()

images = list(set([x.strip() for x in images if x.strip()]))
# 通过私有仓库，将公有镜像下发到内网每台机器上，例如内网ccr.ccs.tencentyun.com的仓库，共约26G
HOST = 'ccr.ccs.tencentyun.com/cube-rancher/'
# print('docker login ')
for image in images:
    # print(image)
    image = image.replace('<none>', '')
    image_name = HOST + image.replace(HOST,'').replace('/', '-').replace('@sha256', '')

    # 可联网机器上拉取公有镜像并推送到私有仓库
    # print('docker pull %s && docker tag %s %s && docker push %s &' % (image,image,image_name,image_name))

    # 内网机器上拉取私有仓库镜像
    # image=image.replace('@sha256','')
    # print("docker pull %s && docker tag %s %s &" % (image_name,image_name,image))

    # 拉取公有镜像
    image=image.replace('@sha256','')
    print("docker pull %s &" % (image,))

print('')
print('wait')






