import { LinkOutlined } from '@ant-design/icons'
import { Button } from 'antd'
import React from 'react'
import { IAppMenuItem } from '../api/interface/kubeflowInterface'

export default function OutLinkTemplate(props?: IAppMenuItem) {
    return (
        <div className="d-f jc ac h100 fade-in">
            <div>
                <div>
                    <img className="pb32 w384" src={require('../images/findData.png')} alt="" />
                    {/* <img className="pb32 w256" src={require('../images/star.svg').default} alt="" /> */}
                </div>
                <div className="ta-c"><Button type="primary" onClick={() => {
                    window.open(props?.url, 'blank')
                }}>{`点击前往${props?.title}`}<LinkOutlined /></Button></div>
            </div>
        </div>
    )
}
