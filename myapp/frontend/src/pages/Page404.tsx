import { Button } from 'antd'
import React from 'react'
import { IAppMenuItem } from '../api/interface/kubeflowInterface'

export default function Page404() {
    return (
        <div className="d-f jc ac h100 fade-in">
            <div className="ta-c">
                <div><img className="w512" src={require('../images/workData.png')} alt=""/></div>
                {/* <div>
                    <img className="pb32 w256" src={require('../images/cube-studio.svg').default} alt="" />
                </div> */}
                <div className="fs16">Wellcome to Cube Studio</div>
            </div>
        </div>
    )
}
