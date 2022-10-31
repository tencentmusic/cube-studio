import { CloseOutlined, DeleteColumnOutlined, DeleteFilled, DeleteOutlined, InboxOutlined, LoadingOutlined, PlusOutlined } from '@ant-design/icons';
import { message } from 'antd';
import Upload, { RcFile, UploadChangeParam } from 'antd/lib/upload';
import { UploadFile } from 'antd/lib/upload/interface';
import React, { useState } from 'react'
import './FileUploadPlus.less';

interface Iprops {
    type?: TFileType
    onChange?: (value: UploadFile[]) => void
    value?: string[]
    maxCount?: number
    maxSize?: number
}

type TFileType = 'file' | 'video' | 'audio'

export default function FileUploadPlus(props: Iprops) {
    const [visableChangePhone, setVisableChangePhone] = useState(false);
    const [fileLoading, setFileLoading] = useState(false);
    const [imgUrl, setImgUrl] = useState('');
    const [imageList, setImageList] = useState<string[]>([])
    const [loading, setLoading] = useState(true);
    const [fileList, setFileList] = useState<UploadFile[]>([])

    function getBase64(img: any, callback: any) {
        const reader = new FileReader();
        reader.addEventListener('load', () => callback(reader.result));
        reader.readAsDataURL(img);
    }

    function beforeUpload(file: RcFile) {
        const maxCount = props.maxCount || 1
        if (fileList.length >= maxCount) {
            message.error('超出文件数量限制');
            return false
        }
        const isFormatOk = true || file.type === 'image/jpeg';
        if (!isFormatOk) {
            message.error('文件格式错误');
        }
        const isLt2M = file.size / 1024 / 1024 < 2;
        if (!isLt2M) {
            message.error('文件大小应小于 2MB');
        }
        return isFormatOk && isLt2M;
    }

    const handleChange = (info: UploadChangeParam) => {
        console.log(info);

        if (info.file.status === 'uploading') {
            setFileLoading(true);
            return;
        }
        if (info.file.status === 'done') {
            setFileLoading(false);
            setFileList(info.fileList)
            props.onChange && props.onChange(info.fileList)
        }
        if (info.file.status === "removed") {
            setFileList(info.fileList)
            props.onChange && props.onChange(info.fileList)
            return;
        }
    };

    //建立一个可存取到该file的url
    function getObjectURL(file: any) {
        var url = null;
        if ((window as any).createObjectURL != undefined) { // basic
            url = (window as any).createObjectURL(file);
        } else if (window.URL != undefined) { // mozilla(firefox)
            url = window.URL.createObjectURL(file);
        } else if (window.webkitURL != undefined) { // webkit or chrome
            url = window.webkitURL.createObjectURL(file);
        }
        return url;
    }

    const createMediaPreview = (file: UploadFile<any>, fileIndex: number, type: TFileType) => {
        const url = getObjectURL(file)
        const key = Math.random().toString(36).substring(2);
        if (type === 'video') {
            return <div className="p-r" key={key}>
                <span
                    onClick={() => {
                        const currentFileList = [...fileList]
                        currentFileList.splice(fileIndex, 1)
                        setFileList(currentFileList)
                    }}
                    className="d-il p-a plr8 ptb2 bg-fail"
                    style={{ top: 0, right: 0, borderBottomLeftRadius: 6, zIndex: 9 }}>
                    <DeleteOutlined style={{ color: '#fff' }} />
                </span>
                <video className="w100 mb8" src={url} controls></video>
            </div>
        } else if (type === 'audio') {
            return <div className="d-f ac mb8" key={key}>
                <audio className="w100 flex1" src={url} controls></audio>
                <span
                    onClick={() => {
                        const currentFileList = [...fileList]
                        currentFileList.splice(fileIndex, 1)
                        setFileList(currentFileList)
                    }}
                    className="d-il plr8 ptb2 bg-fail"
                    style={{ borderRadius: 6 }}>
                    <DeleteOutlined style={{ color: '#fff' }} />
                </span>
            </div>
        }
        return file
    }

    return (
        <>
            <div>
                {
                    fileList.map((file, fileIndex) => {
                        return createMediaPreview(file, fileIndex, 'video')
                    })
                }
            </div>
            <Upload.Dragger
                // name="file"
                fileList={fileList}
                showUploadList={false}
                customRequest={(options) => {
                    console.log(options.file);
                    const tarList = [...fileList, options.file as RcFile]
                    setFileList(tarList)
                    props.onChange && props.onChange(tarList)
                    // getBase64(options.file, (imageUrl: string) => {
                    //     // setImgUrl(imageUrl);
                    //     const tarList = [...imageList, imageUrl]
                    //     setImageList(tarList)
                    //     setFileLoading(false);
                    //     props.onChange && props.onChange(tarList)
                    // });
                }}
                beforeUpload={beforeUpload}
                onChange={handleChange}
            >
                <p className="ant-upload-drag-icon">
                    <InboxOutlined />
                </p>
                <p className="ant-upload-text">点击或拖拽文件上传</p>
            </Upload.Dragger>
        </>
    )
}
