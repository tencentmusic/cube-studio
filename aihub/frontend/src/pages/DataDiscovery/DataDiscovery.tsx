import React, { useState } from 'react'
import DataDiscoverySearch from '../../components/DataDiscoverySearch/DataDiscoverySearch';
import './DataDiscovery.less';

export default function DataDiscovery() {
    const [searchContent, setSearchContent] = useState<string>()

    return (
        <div className="d-f jc ac h100 w100">
            <DataDiscoverySearch
                value={searchContent}
                // isOpenSearchMatch
                onChange={(value) => {
                    setSearchContent(value)
                }}
                placeholder="输入关键字（表名）搜索" />
        </div>
    )
}
