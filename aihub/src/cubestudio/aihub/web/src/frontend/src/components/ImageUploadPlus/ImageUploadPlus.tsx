import { CloseOutlined, LoadingOutlined, PlusOutlined } from '@ant-design/icons';
import { message } from 'antd';
import Upload, { RcFile, UploadChangeParam } from 'antd/lib/upload';
import React, { useState } from 'react'
import './ImageUploadPlus.less';

interface Iprops {
    maxCount?: number
    value?: string[]
    onChange?: (value: string[]) => void
}

export default function ImageUploadPlus(props: Iprops) {
    const [visableChangePhone, setVisableChangePhone] = useState(false);
    const [fileLoading, setFileLoading] = useState(false);
    const [imgUrl, setImgUrl] = useState('');
    const [imageList, setImageList] = useState<string[]>([])
    const [loading, setLoading] = useState(true);

    function getBase64(img: any, callback: any) {
        const reader = new FileReader();
        reader.addEventListener('load', () => callback(reader.result));
        reader.readAsDataURL(img);
    }

    function beforeUpload(file: RcFile) {
        const maxCount = props.maxCount || 1
        if (imageList.length >= maxCount) {
            message.error('超出文件数量限制');
            return false
        }
        const isFormatOk = file.type === 'image/jpeg' || file.type === 'image/png' || file.type === 'image/gif' || file.type === 'image/jpg';
        if (!isFormatOk) {
            message.error('仅支持 JPG/PNG 格式图片');
        }
        const isLt10M = file.size / 1024 / 1024 < 10;
        if (!isLt10M) {
            message.error('图片大小应小于 10MB');
        }
        return isFormatOk && isLt10M;
    }

    const handleChange = (info: UploadChangeParam) => {
        console.log(info);
        if (info.file.status === 'uploading') {
            setFileLoading(true);
            return;
        }
        if (info.file.status === 'done') {
            setFileLoading(false);
        }
    };

    return (
        <div className="d-f fw">
            {
                imageList.map((imageUrl, imageIndex) => {
                    const key = Math.random().toString(36).substring(2);
                    return <div className="image-card s0 mr16 mb16 ov-h" key={key}>
                        <div className="image-close" onClick={() => {
                            const orginList = [...imageList]
                            orginList.splice(imageIndex, 1)
                            setImageList(orginList)
                            props.onChange && props.onChange(orginList)
                        }}>
                            <CloseOutlined style={{ color: '#fff' }} />
                        </div>
                        <img
                            src={imageUrl}
                            alt="file"
                            style={{ width: '100%' }}
                        />
                    </div>
                })
            }
            {
                imageList.length < (props.maxCount || 1) ? <Upload
                    name="file"
                    listType="picture-card"
                    className="file-uploader"
                    showUploadList={false}
                    // action={ }
                    method="POST"
                    customRequest={(options) => {
                        console.log(options.file);
                        getBase64(options.file, (imageUrl: string) => {
                            // setImgUrl(imageUrl);
                            const tarList = [...imageList, imageUrl]
                            setImageList(tarList)
                            setFileLoading(false);
                            props.onChange && props.onChange(tarList)
                        });
                    }}
                    beforeUpload={beforeUpload}
                    onChange={handleChange}
                >
                    <div>
                        {fileLoading ? (
                            <LoadingOutlined />
                        ) : (
                                <PlusOutlined />
                            )}
                        <div style={{ marginTop: 8 }}>上传图片</div>
                    </div>
                </Upload> : null
            }
        </div>
    )
}
