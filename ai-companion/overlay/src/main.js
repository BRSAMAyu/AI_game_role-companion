const { app, BrowserWindow, ipcMain, globalShortcut } = require('electron');
const path = require('path');
const WebSocket = require('ws');

const OVERLAY_PORT = 17865;
let mainWindow = null;
let mouseThrough = true;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 120,
    transparent: true,
    frame: false,
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    focusable: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  mainWindow.setAlwaysOnTop(true, 'screen-saver');
  mainWindow.setIgnoreMouseEvents(mouseThrough, { forward: true });
  mainWindow.loadFile(path.join(__dirname, 'renderer.html'));
}

function registerShortcuts() {
  globalShortcut.register('Alt+`', () => {
    mouseThrough = !mouseThrough;
    if (mainWindow) {
      mainWindow.setIgnoreMouseEvents(mouseThrough, { forward: true });
    }
  });
}

function startWebSocketServer() {
  const wss = new WebSocket.Server({ port: OVERLAY_PORT });
  wss.on('connection', (ws) => {
    ws.on('message', (raw) => {
      try {
        const payload = JSON.parse(raw.toString());
        if (payload.type === 'showText') {
          mainWindow?.webContents.send('show-text', {
            text: payload.text || '',
            ms: payload.ms || 4000,
            style: payload.style || 'default',
          });
        }
      } catch (error) {
        console.error('[Overlay] Failed to process payload', error);
      }
    });
  });
  wss.on('listening', () => {
    console.log(`[Overlay] WebSocket server listening on ws://127.0.0.1:${OVERLAY_PORT}`);
  });
}

app.whenReady().then(() => {
  createWindow();
  registerShortcuts();
  startWebSocketServer();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});

ipcMain.handle('toggle-mouse', (_event, enable) => {
  mouseThrough = Boolean(enable);
  mainWindow?.setIgnoreMouseEvents(mouseThrough, { forward: true });
});
