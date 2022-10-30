import { Form, Input } from 'antd';
import React, { useEffect } from 'react'
import { Outlet, useLocation } from 'react-router-dom';

export default function Home(props?: any) {
    // const location = useLocation()
    // useEffect(() => {
    //     console.log('props', props);
    // }, [props])

    // useEffect(() => {
    //     console.log('location', location);
    // }, [location])

    return (
        <>
            Home
            <Outlet />
        </>
    )
}
