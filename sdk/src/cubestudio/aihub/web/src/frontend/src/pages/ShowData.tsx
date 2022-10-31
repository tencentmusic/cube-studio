import React, { useEffect, useState } from 'react'
import { getParam } from '../util'

export default function ShowData() {
    const [data, setData] = useState<string>('')
    useEffect(() => {
        // some api get data
    }, [])
    return (
        <div className="fade-in">
            test
        </div>
    )
}
