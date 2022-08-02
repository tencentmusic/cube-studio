import React, { useState } from 'react'
import { IAppMenuItem } from '../api/interface/kubeflowInterface'
import { getParam } from '../util'

export default function IframeTemplate(props?: IAppMenuItem) {
    const [url, setUrl] = useState(getParam('url') || props?.url)
    return (
        <>
            <iframe id="_frontendAppCustomFrame_"
                // src="http://kubeflow.tke.woa.com/static/appbuilder/vison/index.html#/HeterogeneousPlatform/Nationalkaraoke"
                src={url}
                allowFullScreen
                allow="microphone;camera;midi;encrypted-media;"
                className="w100 h100 fade-in"
                style={{ border: 0 }}>
            </iframe>
        </>
    )
}
