import { IStackStyles, IStackItemStyles, mergeStyles } from '@fluentui/react';

const headStyle: Partial<IStackStyles> = {
  root: {
    backgroundColor: '#fff',
    padding: '10px 16px',
    borderBottom: '1px solid #dadada',
  },
};
const headNameStyle = mergeStyles({
  boxSizing: 'border-box',
  marginLeft: '8px',
  padding: '0 8px',
  height: '32px',
  lineHeight: '32px',
  cursor: 'text',
  marginBottom: '4px',
  fontSize: '16px',
  fontWeight: 600,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
  color: 'black',
});
const buttonItemStyle: Partial<IStackItemStyles> = {
  root: {
    marginRight: '16px',
  },
};

const hidden = mergeStyles({
  visibility: 'hidden',
});

export default {
  headStyle,
  headNameStyle,
  buttonItemStyle,
  hidden,
};
