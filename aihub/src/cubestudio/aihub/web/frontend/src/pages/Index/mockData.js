export const data = {
    "aihub_url": "/frontend/aihub/model_market/model_visual",
    "describe": "ai\u793a\u4f8b\u5e94\u7528\uff0c\u8be6\u7ec6\u63cf\u8ff0\uff0c\u90fd\u4f1a\u663e\u793a\u5e94\u7528\u63cf\u8ff0\u4e0a\uff0c\u652f\u6301markdown",
    "doc": "https://github.com/tencentmusic/cube-studio/tree/master/aihub",
    "field": "\u673a\u5668\u89c6\u89c9",
    "github_url": "https://github.com/tencentmusic/cube-studio",
    "inference_inputs": [
        {
            "choices": [],
            "default": "\u8fd9\u91cc\u662f\u9ed8\u8ba4\u503c",
            "describe": "arg1\u7684\u8be6\u7ec6\u8bf4\u660e\uff0c\u7528\u4e8e\u5728\u754c\u9762\u5c55\u793a",
            "label": "\u63a8\u7406\u51fd\u6570\u7684\u8f93\u5165\u53c2\u6570arg1",
            "name": "arg1",
            "type": "text",
            "validators": [
                {
                    "max": -1,
                    "min": -1,
                    "regex": "[a-z]*",
                    "type": "Regexp"
                }
            ]
        },
        {
            "choices": [],
            "default": "",
            "describe": "arg2\u7684\u8be6\u7ec6\u8bf4\u660e\uff0c\u7528\u4e8e\u5728\u754c\u9762\u5c55\u793a,\u4f20\u9012\u5230\u63a8\u7406\u51fd\u6570\u4e2d\u5c06\u662f\u56fe\u7247\u672c\u5730\u5730\u5740",
            "label": "\u63a8\u7406\u51fd\u6570\u7684\u8f93\u5165\u53c2\u6570arg2",
            "name": "arg2",
            "type": "image",
            "validators": [
                {
                    "max": 2,
                    "min": -1,
                    "regex": "",
                    "type": "Length"
                },
                {
                    "max": -1,
                    "min": -1,
                    "regex": "",
                    "type": "DataRequired"
                }
            ]
        },
        {
            "choices": [],
            "default": "",
            "describe": "arg3\u7684\u8be6\u7ec6\u8bf4\u660e\uff0c\u7528\u4e8e\u5728\u754c\u9762\u5c55\u793a,\u4f20\u9012\u5230\u63a8\u7406\u51fd\u6570\u4e2d\u5c06\u662f\u89c6\u9891\u672c\u5730\u5730\u5740",
            "label": "\u63a8\u7406\u51fd\u6570\u7684\u8f93\u5165\u53c2\u6570arg3",
            "name": "arg3",
            "type": "video",
            "validators": []
        },
        {
            "choices": [],
            "default": "",
            "describe": "arg4\u7684\u8be6\u7ec6\u8bf4\u660e\uff0c\u7528\u4e8e\u5728\u754c\u9762\u5c55\u793a,\u4f20\u9012\u5230\u63a8\u7406\u51fd\u6570\u4e2d\u5c06\u662f\u56fe\u7247\u672c\u5730\u5730\u5740",
            "label": "\u63a8\u7406\u51fd\u6570\u7684\u8f93\u5165\u53c2\u6570arg4",
            "name": "arg4",
            "type": "image_multi",
            "validators": []
        },
        {
            "choices": [],
            "default": "",
            "describe": "arg5\u7684\u8be6\u7ec6\u8bf4\u660e\uff0c\u7528\u4e8e\u5728\u754c\u9762\u5c55\u793a,\u4f20\u9012\u5230\u63a8\u7406\u51fd\u6570\u4e2d\u5c06\u662f\u89c6\u9891\u672c\u5730\u5730\u5740",
            "label": "\u63a8\u7406\u51fd\u6570\u7684\u8f93\u5165\u53c2\u6570arg5",
            "name": "arg5",
            "type": "video_multi",
            "validators": []
        },
        {
            "choices": [
                "choice1",
                "choice2",
                "choice3"
            ],
            "default": "choice2",
            "describe": "arg6\u7684\u8be6\u7ec6\u8bf4\u660e\uff0c\u7528\u4e8e\u5728\u754c\u9762\u5c55\u793a,\u5355\u9009\u7ec4\u4ef6",
            "label": "\u63a8\u7406\u51fd\u6570\u7684\u8f93\u5165\u53c2\u6570arg6",
            "name": "arg6",
            "type": "text_select",
            "validators": []
        },
        {
            "choices": [
                "http://localhost:8080/app1/static/example/\u98ce\u683c1.jpg",
                "http://localhost:8080/app1/static/example/\u98ce\u683c2.jpg"
            ],
            "default": "http://localhost:8080/app1/static/example/\u98ce\u683c2.jpg",
            "describe": "arg7\u7684\u8be6\u7ec6\u8bf4\u660e\uff0c\u7528\u4e8e\u5728\u754c\u9762\u5c55\u793a,\u591a\u9009\u7ec4\u4ef6",
            "label": "\u63a8\u7406\u51fd\u6570\u7684\u8f93\u5165\u53c2\u6570arg7",
            "name": "arg7",
            "type": "image_select",
            "validators": []
        },
        {
            "choices": [
                "choice1",
                "choice2",
                "choice3"
            ],
            "default": [
                "choice1"
            ],
            "describe": "arg8\u7684\u8be6\u7ec6\u8bf4\u660e\uff0c\u7528\u4e8e\u5728\u754c\u9762\u5c55\u793a,\u591a\u9009\u7ec4\u4ef6",
            "label": "\u63a8\u7406\u51fd\u6570\u7684\u8f93\u5165\u53c2\u6570arg8",
            "name": "arg8",
            "type": "text_select_multi",
            "validators": []
        },
        {
            "choices": [
                "http://localhost:8080/app1/static/example/\u98ce\u683c1.jpg",
                "http://localhost:8080/app1/static/example/\u98ce\u683c2.jpg"
            ],
            "default": [
                "http://localhost:8080/app1/static/example/\u98ce\u683c2.jpg",
                "http://localhost:8080/app1/static/example/\u98ce\u683c2.jpg"
            ],
            "describe": "arg9\u7684\u8be6\u7ec6\u8bf4\u660e\uff0c\u7528\u4e8e\u5728\u754c\u9762\u5c55\u793a,\u591a\u9009\u7ec4\u4ef6",
            "label": "\u63a8\u7406\u51fd\u6570\u7684\u8f93\u5165\u53c2\u6570arg9",
            "name": "arg9",
            "type": "image_select_multi",
            "validators": [
                {
                    "max": 2,
                    "min": -1,
                    "regex": "",
                    "type": "Length"
                }
            ]
        }
    ],
    "inference_url": "/app1/api/model/app1/version/v20221001/",
    "label": "\u793a\u4f8b\u5e94\u7528\u4e2d\u6587\u540d",
    "name": "app1",
    "pic": "https://user-images.githubusercontent.com/20157705/170216784-91ac86f7-d272-4940-a285-0c27d6f6cd96.jpg",
    "rec_apps": [
        {
            "label": "\u56fe\u7247\u4fee\u590d",
            "pic": "https://p6.toutiaoimg.com/origin/tos-cn-i-qvj2lq49k0/6a284d35f42b414d9f4dcb474b0e644f"
        },
        {
            "label": "\u56fe\u7247\u4fee\u590d",
            "pic": "https://p6.toutiaoimg.com/origin/tos-cn-i-qvj2lq49k0/6a284d35f42b414d9f4dcb474b0e644f"
        },
        {
            "label": "\u56fe\u7247\u4fee\u590d",
            "pic": "https://p6.toutiaoimg.com/origin/tos-cn-i-qvj2lq49k0/6a284d35f42b414d9f4dcb474b0e644f"
        }
    ],
    "scenes": "\u56fe\u50cf\u8bc6\u522b",
    "status": "online",
    "user": "/app1/login",
    "version": "v20221001",
    "web_examples": [
        {
            "arg1": "\u6d4b\u8bd5\u8f93\u5165\u6587\u672c",
            "arg2": "http://localhost:8080/app1/static/example/test.jpg",
            "arg3": "https://pengluan-76009.sz.gfp.tencent-cloud.com/cube-studio%20install.mp4"
        }
    ]
}