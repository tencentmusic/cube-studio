import { IIconProps, mergeStyleSets, mergeStyles } from '@fluentui/react';
const modalStyles = mergeStyles({
  position: 'fixed',
  width: '100%',
  height: '100vh',
  backgroundColor: 'rgba(0, 0, 0, .5)',
  zIndex: 999,
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
});
const cancelIcon: IIconProps = { iconName: 'Cancel' };
const contentStyles = mergeStyleSets({
  container: {
    minWidth: '75%',
    minHeight: '80%',
    overflow: 'hidden',
    display: 'flex',
    borderRadius: 2,
    backgroundColor: '#fff',
    padding: '0 24px',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '10px 0 14px 0',
  },
  body: {
    flex: 1,
    height: 0,
    position: 'relative',
    display: 'flex',
  },
  footer: {
    display: 'flex',
    padding: '24px 0',
    justifyContent: 'flex-end',
  },
});
const resizeLine = mergeStyles({
  width: 8,
  height: '100%',
  cursor: 'col-resize',
  background: '#eee',
  transition: 'background 0.2s',
  margin: '0 15px 0 2px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  overflow: 'hidden',
});
const resizeLineIconStyles = mergeStyles({
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  boxSizing: 'border-box',
  paddingRight: '19px',
  width: 8,
  height: 20,
  fontSize: 20,
  pointerEvents: 'none',
});

export default {
  modalStyles,
  cancelIcon,
  contentStyles,
  resizeLine,
  resizeLineIconStyles,
};
