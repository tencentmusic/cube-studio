import { mergeStyles } from '@fluentui/react';

const itemStyles = mergeStyles({
  position: 'relative',
});
const flowWrapper = mergeStyles({
  width: '100%',
  height: '100%',
});
const spinnerWrapper = mergeStyles({
  position: 'absolute',
  top: 0,
  left: 0,
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  width: '100%',
  height: '100%',
  overflow: 'hidden',
  backgroundColor: 'rgba(255, 255, 255, 0.6)',
  zIndex: 9999,
});

export default {
  itemStyles,
  flowWrapper,
  spinnerWrapper,
};
