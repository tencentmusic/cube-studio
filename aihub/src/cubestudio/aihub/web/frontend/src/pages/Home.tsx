import { Form, Input } from 'antd';
import React, { useEffect } from 'react'
import { Outlet, useLocation } from 'react-router-dom';

export default function Home(props?: any) {

    return (
        <>
            <Outlet />
        </>
    )
}
