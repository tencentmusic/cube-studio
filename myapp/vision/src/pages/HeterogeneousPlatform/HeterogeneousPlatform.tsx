import React, { useState, useEffect } from 'react';
import { Card } from 'antd';
import { useHistory } from 'react-router-dom';
import './HeterogeneousPlatform.css';
function HeterogeneousPlatform() {
  const history = useHistory();
  const NationalKaraokeRouter = () => {
    history.push({
      pathname: `/HeterogeneousPlatform/Nationalkaraoke`,
      state: 2,
    });
  };
  const QQNationalKaraokeRouter = () => {
    history.push({
      pathname: `/HeterogeneousPlatform/Nationalkaraoke`,
      state: 1,
    });
  };
  const RegistrationOperatorRouter = () => {
    history.push('/HeterogeneousPlatform/Nationalkaraoke/RegistrationOperator');
  };
  return (
    <div className="HeterogeneousPlatformClass">
      <Card hoverable style={{ width: 300 }} onClick={NationalKaraokeRouter}>
        <h1>全民K歌</h1>
      </Card>
      <Card hoverable style={{ width: 300, marginLeft: 20 }} onClick={QQNationalKaraokeRouter}>
        <h1>QQ音乐</h1>
      </Card>
      <Card hoverable style={{ width: 300, marginLeft: 20 }} onClick={RegistrationOperatorRouter}>
        <h1>注册算子</h1>
      </Card>
    </div>
  );
}
export default HeterogeneousPlatform;
