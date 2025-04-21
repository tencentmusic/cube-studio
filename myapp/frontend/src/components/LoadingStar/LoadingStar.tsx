import React from 'react'
import globalConfig from '../../global.config';
import './LoadingStar.less';

export default function LoadingStar() {
    return (
        <>
            <img className="loading-cb" src={globalConfig.loadingLogo.default} alt="" />
        </>
    )
}
