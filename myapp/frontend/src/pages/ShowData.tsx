import React, { useEffect, useState } from 'react'
import { getParam } from '../util'

export default function ShowData() {
    const [data, setData] = useState<string>('')
    useEffect(() => {
        // some api get data
    }, [])
    return (
        <div className="fade-in" dangerouslySetInnerHTML={{ __html: data }}>

        </div>
    )
}
