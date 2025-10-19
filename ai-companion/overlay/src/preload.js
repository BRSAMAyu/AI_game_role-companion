const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('overlayAPI', {
  onShowText: (callback) => {
    ipcRenderer.on('show-text', (_event, payload) => callback(payload));
  },
  toggleMouse: (enabled) => ipcRenderer.invoke('toggle-mouse', enabled),
});
