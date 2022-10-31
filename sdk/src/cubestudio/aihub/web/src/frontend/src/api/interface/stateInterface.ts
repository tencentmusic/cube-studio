export interface IExampleItem {
    "img_file_path": string
}

export interface IInputItem {
    "choices": [],
    "default": string,
    "describe": string,
    "label": string,
    "name": string,
    "type": string,
    "validators": []
}

export interface IRecAppItem {
    label: string
    pic: string
}

export interface IAppInfo {
    "describe": string,
    "doc": string,
    "field": string,
    "inference_inputs": IInputItem[],
    "label": string,
    "name": string,
    "pic": string,
    "scenes": string,
    "status": string,
    "version": string,
    "web_examples": IExampleItem[]
    rec_apps: IRecAppItem[]
    inference_url: string
}

export interface IResultItem {
    image: string;
    text: string;
    video: string
}
