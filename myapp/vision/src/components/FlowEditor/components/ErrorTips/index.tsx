import React, { useEffect } from 'react';
import { MessageBar, MessageBarType } from '@fluentui/react';
import { useAppSelector, useAppDispatch } from '../../../../models/hooks';
import { selectErrMsg, updateErrMsg } from '../../../../models/app';

const ErrorTips: React.FC = () => {
  const dispatch = useAppDispatch();
  const errMsg = useAppSelector(selectErrMsg);

  // tips 3s 后隐藏
  useEffect(() => {
    if (errMsg) {
      setTimeout(() => {
        dispatch(updateErrMsg(null));
      }, 3000);
    }
  }, [errMsg]);

  return (
    errMsg && (
      <MessageBar
        messageBarType={MessageBarType.error}
        styles={{
          root: {
            minHeight: 32,
            width: '80%',
            margin: '0 auto',
          },
        }}
        dismissButtonAriaLabel="Close"
        onDismiss={() => {
          dispatch(updateErrMsg(null));
        }}
        isMultiline={false}
      >
        {errMsg?.msg || ''}
      </MessageBar>
    )
  );
};

export default ErrorTips;
