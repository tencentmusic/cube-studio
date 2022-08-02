import { mergeStyles } from '@fluentui/react';

const calloutContent = mergeStyles({
  display: 'flex',
  flexFlow: 'column nowrap',
  width: '360px',
  height: 'auto',
  boxSizing: 'border-box',
  maxHeight: 'inherit',
  padding: '16px 0px',
  fontSize: '12px',
  background: '#fff',
  borderRadius: '2px',
});

const moduleDetailItemStyle = mergeStyles({
  fontSize: '12px',
  padding: '0 16px',
  overflowX: 'visible',
  overflowY: 'hidden',
});

const moduleDetailTitle = mergeStyles({
  padding: '0px 16px',
  fontWeight: 600,
  fontSize: '14px',
  lineHeight: '1',
  marginTop: '0px',
  marginBottom: '8px',
});

const moduleDetailLabel = mergeStyles({
  fontSize: '12px',
  fontWeight: 'bold',
  lineHeight: '16px',
  color: 'rgb(89, 89, 89)',
  padding: '0px',
  margin: '0px 0px 4px',
});
const moduleDetailBody = mergeStyles({
  lineHeight: '14px',
  marginBottom: '22px',
  p: {
    margin: '0',
    padding: '0',
  },
});
const moduleButton = mergeStyles({
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
});

export default {
  calloutContent,
  moduleDetailItemStyle,
  moduleDetailTitle,
  moduleDetailLabel,
  moduleDetailBody,
  moduleButton,
};
